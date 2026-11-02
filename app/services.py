"""
SEQUORA — Services
=====================
Loads every trained model ONCE at startup and exposes the actual
recommendation logic that app/main.py's endpoints call.

Fallback behavior: if SASRec/Two-Tower/FAISS fail or are simply not
available (e.g. you haven't trained them on this machine yet), recommend()
falls back to a precomputed popularity ranking rather than raising an
error — this matches the "graceful degradation" requirement from the
project spec.

v2: the fallback now prefers LAST-WEEK popularity (processed_data/
recent_popularity.json), which beat all-time popularity ~2.7x in testing,
and only then falls back to all-time popularity.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from app.catalog import build_item_categories
from app.schemas import RecommendedItem
from models.sasrec import SASRec
from models.two_tower import TwoTower
from retrieval.faiss_index import FaissItemIndex

logger = logging.getLogger("sequora.services")


def pad_left(seq: List[int], max_len: int) -> List[int]:
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


class RecommenderService:
    """
    One instance of this is created at FastAPI startup (see app/main.py)
    and reused across every request. Holds all loaded models plus a tiny
    in-memory session store for cold-start users (click history that
    hasn't been persisted to a real user account yet).
    """

    def __init__(
        self,
        processed_dir: Path,
        artifacts_dir: Path,
        max_len: int = 50,
    ) -> None:
        self.processed_dir = processed_dir
        self.artifacts_dir = artifacts_dir
        self.max_len = max_len

        self.sasrec: Optional[SASRec] = None
        self.two_tower: Optional[TwoTower] = None
        self.faiss_index: Optional[FaissItemIndex] = None
        self.popularity_ranking: List[int] = []
        self.recent_popularity_ranking: List[int] = []
        self.item_categories: List[Optional[str]] = []
        self.train_sequences: Dict[int, Dict[str, List]] = {}
        self.content_matrix: Optional[np.ndarray] = None

        # session_id -> list of item ids clicked this session (cold-start users)
        self.sessions: Dict[str, List[int]] = {}

        self._load_everything()

    # ------------------------------------------------------------------ #
    # Startup loading — each piece is independently optional, so a
    # missing file degrades gracefully instead of crashing the service.
    # ------------------------------------------------------------------ #

    def _load_everything(self) -> None:
        self._load_sequences_and_popularity()
        self._load_recent_popularity()
        self._load_item_categories()
        self._load_sasrec()
        self._load_two_tower_and_faiss()
        logger.info(
            "RecommenderService ready. sasrec=%s two_tower=%s faiss_size=%s",
            self.sasrec is not None, self.two_tower is not None,
            self.faiss_index.size if self.faiss_index else 0,
        )

    def _load_sequences_and_popularity(self) -> None:
        import pickle
        seq_path = self.processed_dir / "sequences_train.pkl"
        if not seq_path.exists():
            logger.warning("No train sequences found at %s — service will have no personalization.", seq_path)
            return
        with open(seq_path, "rb") as f:
            self.train_sequences = pickle.load(f)

        counts: Dict[int, int] = {}
        for data in self.train_sequences.values():
            for item in data["items"]:
                counts[item] = counts.get(item, 0) + 1
        self.popularity_ranking = sorted(counts, key=counts.get, reverse=True)
        logger.info("Loaded %d user sequences, popularity ranking of %d items",
                    len(self.train_sequences), len(self.popularity_ranking))

    def _load_recent_popularity(self) -> None:
        path = self.processed_dir / "recent_popularity.json"
        if not path.exists():
            logger.warning("No recent_popularity.json at %s — using all-time popularity.", path)
            return
        self.recent_popularity_ranking = [int(i) for i in json.loads(path.read_text())]
        logger.info("Loaded recent popularity ranking of %d items", len(self.recent_popularity_ranking))

    def _load_item_categories(self) -> None:
        try:
            self.item_categories = build_item_categories(self.processed_dir)
            logger.info("Built category map for %d items", len(self.item_categories))
        except Exception:
            logger.exception("Could not build item categories — category filtering disabled.")

    def _in_category(self, item_id: int, category: Optional[str]) -> bool:
        if not category or not self.item_categories:
            return True
        return 0 <= item_id < len(self.item_categories) and self.item_categories[item_id] == category

    def _load_sasrec(self) -> None:
        path = self.artifacts_dir / "sasrec.pt"
        if not path.exists():
            logger.warning("SASRec checkpoint not found at %s — sequential engine disabled.", path)
            return
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model = SASRec(
                num_items=ckpt["num_items"], max_len=ckpt["max_len"],
                d_model=ckpt["d_model"], num_heads=ckpt["num_heads"],
                num_blocks=ckpt["num_blocks"], dropout=ckpt["dropout"],
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.sasrec = model
        except Exception:
            logger.exception("Failed to load SASRec checkpoint — sequential engine disabled.")

    def _load_two_tower_and_faiss(self) -> None:
        tt_path = self.artifacts_dir / "two_tower.pt"
        faiss_path = self.artifacts_dir / "faiss_items.index"
        if not (tt_path.exists() and faiss_path.exists()):
            logger.warning("Two-Tower checkpoint or FAISS index missing — retrieval engine disabled.")
            return
        try:
            ckpt = torch.load(tt_path, map_location="cpu", weights_only=False)
            model = TwoTower(
                num_items=ckpt["num_items"], content_dim=ckpt["content_dim"],
                item_emb_dim=ckpt["item_emb_dim"], output_dim=ckpt["output_dim"],
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.two_tower = model
            self.faiss_index = FaissItemIndex.load(faiss_path, dim=ckpt["output_dim"])
        except Exception:
            logger.exception("Failed to load Two-Tower/FAISS — retrieval engine disabled.")

    # ------------------------------------------------------------------ #
    # Session handling (cold-start users)
    # ------------------------------------------------------------------ #

    def record_event(self, session_id: str, item_id: int) -> int:
        self.sessions.setdefault(session_id, []).append(item_id)
        return len(self.sessions[session_id])

    def get_session_items(self, session_id: str) -> List[int]:
        return self.sessions.get(session_id, [])

    # ------------------------------------------------------------------ #
    # Recommendation logic, with fallback
    # ------------------------------------------------------------------ #

    def _popularity_fallback(
        self, exclude: List[int], top_k: int, category: Optional[str] = None
    ) -> List[RecommendedItem]:
        picked: List[int] = []
        seen = set(exclude)
        # last-week popularity first, then all-time popularity to fill the rest
        for i in self.recent_popularity_ranking + self.popularity_ranking:
            if i not in seen and self._in_category(i, category):
                picked.append(i)
                seen.add(i)
                if len(picked) >= top_k:
                    break
        return [
            RecommendedItem(item_id=item_id, score=0.0, source="popularity_fallback")
            for item_id in picked
        ]

    def recommend(
        self,
        user_id: Optional[int],
        session_item_ids: List[int],
        top_k: int,
        category: Optional[str] = None,
        timeout_seconds: float = 2.0,
    ) -> List[RecommendedItem]:
        """
        Priority order:
          1. If the user has train-time history -> SASRec (sequence-aware)
          2. Else if there's session activity (new user, some clicks) ->
             Two-Tower encoded from the session as a makeshift sequence
          3. Else -> popularity fallback
        If `category` is given, a much deeper list is fetched first and then
        filtered, so the category fills a full grid.
        Any exception along the way falls back to popularity rather than
        propagating an error to the caller.
        """
        start = time.perf_counter()
        history = list(self.train_sequences.get(user_id, {}).get("items", [])) if user_id else []
        combined_history = history + session_item_ids
        fetch_k = top_k if not category else min(max(top_k * 15, 1500), 5000)

        try:
            if combined_history and self.sasrec is not None:
                input_seq = torch.tensor(
                    [pad_left(combined_history, self.max_len)], dtype=torch.long
                )
                if (time.perf_counter() - start) > timeout_seconds:
                    raise TimeoutError("SASRec exceeded latency budget")
                top_items = self.sasrec.recommend(input_seq, k=fetch_k, exclude_seen=True)
                ids = [int(i) for i in top_items[0].tolist() if self._in_category(int(i), category)][:top_k]
                if ids:
                    return [RecommendedItem(item_id=i, score=1.0, source="sasrec") for i in ids]

            if combined_history and self.two_tower is not None and self.faiss_index is not None:
                input_seq = torch.tensor(
                    [pad_left(combined_history, self.max_len)], dtype=torch.long
                )
                with torch.no_grad():
                    user_vec = self.two_tower.encode_users(input_seq).numpy()
                if (time.perf_counter() - start) > timeout_seconds:
                    raise TimeoutError("Two-Tower/FAISS exceeded latency budget")
                scores, result_ids = self.faiss_index.search(user_vec, k=fetch_k + len(combined_history))
                seen = set(combined_history)
                items = [
                    (int(i), float(s)) for i, s in zip(result_ids[0], scores[0])
                    if i != -1 and int(i) not in seen and self._in_category(int(i), category)
                ][:top_k]
                if items:
                    return [
                        RecommendedItem(item_id=item_id, score=score, source="two_tower")
                        for item_id, score in items
                    ]

        except Exception:
            logger.exception("Recommendation engine failed — falling back to popularity.")

        return self._popularity_fallback(exclude=combined_history, top_k=top_k, category=category)

    # ------------------------------------------------------------------ #
    # Health / diagnostics
    # ------------------------------------------------------------------ #

    def health(self) -> dict:
        return {
            "status": "ok",
            "sasrec_loaded": self.sasrec is not None,
            "two_tower_loaded": self.two_tower is not None,
            "faiss_index_size": self.faiss_index.size if self.faiss_index else 0,
            "ranker_loaded": False,  # wired in once app/main.py loads the ranker too
        }