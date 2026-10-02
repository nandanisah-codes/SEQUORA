"""
SEQUORA — Evaluation Suite
=============================
Compares every model on the SAME metric (Recall@10 / NDCG@10, same
evaluation users, same evaluation logic) so the numbers are genuinely
comparable:

  Popularity   — non-personalized baseline: everyone gets the same
                 most-popular items (minus their own history)
  ALS          — classical collaborative filtering baseline
  SASRec       — sequential engine
  Two-Tower    — retrieval engine (via the real FAISS index)
  Cold-start   — Two-Tower's accuracy specifically on items that had
                 ZERO training interactions

Every model is wrapped as a `recommend_fn(user_id) -> list[item_id]`, so
one evaluation function handles all of them identically — no model gets
an easier or harder test than another.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from data.dataset import load_sequences
from models.sasrec import SASRec
from models.two_tower import TwoTower
from retrieval.faiss_index import FaissItemIndex

logger = logging.getLogger("sequora.evaluate")

RecommendFn = Callable[[int], List[int]]


def pad_left(seq: List[int], max_len: int) -> List[int]:
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


# --------------------------------------------------------------------------- #
# Generic evaluation — every model plugs into this the same way
# --------------------------------------------------------------------------- #

def evaluate_recommender(
    name: str,
    recommend_fn: RecommendFn,
    train_sequences: Dict[int, Dict[str, List]],
    val_sequences: Dict[int, Dict[str, List]],
    k: int,
    max_eval_users: int,
    seed: int,
    eligible_user_ids: Optional[List[int]] = None,
) -> Dict[str, float]:
    if eligible_user_ids is None:
        eligible_user_ids = [uid for uid in val_sequences if uid in train_sequences]

    if not eligible_user_ids:
        logger.warning("[%s] No eligible users to evaluate.", name)
        return {"model": name, "recall_at_k": 0.0, "ndcg_at_k": 0.0, "n_eval_users": 0}

    rng = random.Random(seed)
    sampled = (
        rng.sample(eligible_user_ids, max_eval_users)
        if len(eligible_user_ids) > max_eval_users
        else eligible_user_ids
    )

    hits = 0
    ndcg_sum = 0.0
    for user_id in sampled:
        target = val_sequences[user_id]["items"][0]
        ranked = recommend_fn(user_id)
        if target in ranked:
            hits += 1
            rank = ranked.index(target) + 1
            ndcg_sum += 1.0 / np.log2(rank + 1)

    n = len(sampled)
    metrics = {
        "model": name,
        "recall_at_k": hits / n,
        "ndcg_at_k": ndcg_sum / n,
        "n_eval_users": n,
    }
    logger.info(
        "[%s] Recall@%d = %.4f, NDCG@%d = %.4f (n=%d users)",
        name, k, metrics["recall_at_k"], k, metrics["ndcg_at_k"], n,
    )
    return metrics


# --------------------------------------------------------------------------- #
# Popularity baseline
# --------------------------------------------------------------------------- #

def build_popularity_ranking(train_sequences: Dict[int, Dict[str, List]]) -> List[int]:
    counts: Dict[int, int] = {}
    for data in train_sequences.values():
        for item in data["items"]:
            counts[item] = counts.get(item, 0) + 1
    ranked = sorted(counts, key=counts.get, reverse=True)
    return ranked


def make_popularity_recommend_fn(
    train_sequences: Dict[int, Dict[str, List]], popularity_ranking: List[int], k: int
) -> RecommendFn:
    def recommend(user_id: int) -> List[int]:
        seen = set(train_sequences.get(user_id, {}).get("items", []))
        result = [item for item in popularity_ranking if item not in seen]
        return result[:k]
    return recommend


# --------------------------------------------------------------------------- #
# ALS baseline
# --------------------------------------------------------------------------- #

def train_als(
    train_sequences: Dict[int, Dict[str, List]], num_items: int, factors: int, seed: int
):
    from implicit.als import AlternatingLeastSquares

    user_ids = sorted(train_sequences.keys())
    user_idx_map = {uid: i for i, uid in enumerate(user_ids)}

    rows, cols, data = [], [], []
    for uid, seq in train_sequences.items():
        u = user_idx_map[uid]
        for item in seq["items"]:
            rows.append(u)
            cols.append(item)
            data.append(1.0)

    matrix = sp.csr_matrix(
        (data, (rows, cols)), shape=(len(user_ids), num_items + 1)
    )

    model = AlternatingLeastSquares(factors=factors, iterations=15, random_state=seed)
    model.fit(matrix)
    logger.info("Trained ALS with %d factors on %d users, %d items", factors, len(user_ids), num_items)
    return model, matrix, user_idx_map


def make_als_recommend_fn(model, matrix, user_idx_map: Dict[int, int], k: int) -> RecommendFn:
    def recommend(user_id: int) -> List[int]:
        if user_id not in user_idx_map:
            return []
        u = user_idx_map[user_id]
        item_ids, _scores = model.recommend(
            u, matrix[u], N=k, filter_already_liked_items=True
        )
        return [int(i) for i in item_ids]
    return recommend


# --------------------------------------------------------------------------- #
# SASRec
# --------------------------------------------------------------------------- #

def load_sasrec(checkpoint_path: Path) -> SASRec:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SASRec(
        num_items=ckpt["num_items"],
        max_len=ckpt["max_len"],
        d_model=ckpt["d_model"],
        num_heads=ckpt["num_heads"],
        num_blocks=ckpt["num_blocks"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def make_sasrec_recommend_fn(
    model: SASRec, train_sequences: Dict[int, Dict[str, List]], max_len: int, k: int
) -> RecommendFn:
    def recommend(user_id: int) -> List[int]:
        items = train_sequences.get(user_id, {}).get("items", [])
        input_seq = torch.tensor([pad_left(items, max_len)], dtype=torch.long)
        top_items = model.recommend(input_seq, k=k, exclude_seen=True)
        return top_items[0].tolist()
    return recommend


# --------------------------------------------------------------------------- #
# Two-Tower + FAISS
# --------------------------------------------------------------------------- #

def load_two_tower(checkpoint_path: Path) -> TwoTower:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TwoTower(
        num_items=ckpt["num_items"],
        content_dim=ckpt["content_dim"],
        item_emb_dim=ckpt["item_emb_dim"],
        output_dim=ckpt["output_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def make_two_tower_recommend_fn(
    model: TwoTower,
    index: FaissItemIndex,
    train_sequences: Dict[int, Dict[str, List]],
    max_len: int,
    k: int,
) -> RecommendFn:
    def recommend(user_id: int) -> List[int]:
        items = train_sequences.get(user_id, {}).get("items", [])
        input_seq = torch.tensor([pad_left(items, max_len)], dtype=torch.long)
        with torch.no_grad():
            user_vec = model.encode_users(input_seq).numpy()
        _scores, result_ids = index.search(user_vec, k=k + len(items))
        seen = set(items)
        filtered = [int(i) for i in result_ids[0] if i != -1 and int(i) not in seen]
        return filtered[:k]
    return recommend


# --------------------------------------------------------------------------- #
# Cold-start evaluation (Two-Tower only, restricted to items with real
# multimodal content vectors AND zero train interactions)
# --------------------------------------------------------------------------- #

def evaluate_cold_start(
    recommend_fn: RecommendFn,
    cold_item_ids: List[int],
    val_sequences: Dict[int, Dict[str, List]],
    train_sequences: Dict[int, Dict[str, List]],
    k: int,
    max_eval_users: int,
    seed: int,
) -> Dict[str, float]:
    cold_set = set(cold_item_ids)
    eligible = [
        uid for uid, seq in val_sequences.items()
        if uid in train_sequences and seq["items"][0] in cold_set
    ]
    if not eligible:
        logger.warning(
            "No validation users have a cold-start item as their next interaction — "
            "cold-start evaluation skipped (expand multimodal.py's --limit to cover more items)."
        )
        return {"model": "two_tower_cold_start", "recall_at_k": None, "ndcg_at_k": None, "n_eval_users": 0}

    return evaluate_recommender(
        "two_tower_cold_start", recommend_fn, train_sequences, val_sequences,
        k=k, max_eval_users=max_eval_users, seed=seed, eligible_user_ids=eligible,
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run(args: argparse.Namespace) -> None:
    article_id_map = json.loads((args.processed_dir / "article_id_map.json").read_text())
    num_items = max(article_id_map.values())

    train_sequences = load_sequences(args.processed_dir / "sequences_train.pkl")
    val_sequences = load_sequences(args.processed_dir / "sequences_val.pkl")
    cold_start_items = json.loads((args.processed_dir / "cold_start_items.json").read_text())

    results = []

    # 1. Popularity
    popularity_ranking = build_popularity_ranking(train_sequences)
    pop_fn = make_popularity_recommend_fn(train_sequences, popularity_ranking, args.eval_k)
    results.append(evaluate_recommender(
        "popularity", pop_fn, train_sequences, val_sequences,
        args.eval_k, args.eval_users, args.seed,
    ))

    # 2. ALS
    als_model, als_matrix, user_idx_map = train_als(
        train_sequences, num_items, factors=args.als_factors, seed=args.seed
    )
    als_fn = make_als_recommend_fn(als_model, als_matrix, user_idx_map, args.eval_k)
    results.append(evaluate_recommender(
        "als", als_fn, train_sequences, val_sequences,
        args.eval_k, args.eval_users, args.seed,
    ))

    # 3. SASRec
    sasrec_path = args.artifacts_dir / "sasrec.pt"
    if sasrec_path.exists():
        sasrec_model = load_sasrec(sasrec_path)
        sasrec_fn = make_sasrec_recommend_fn(sasrec_model, train_sequences, args.max_len, args.eval_k)
        results.append(evaluate_recommender(
            "sasrec", sasrec_fn, train_sequences, val_sequences,
            args.eval_k, args.eval_users, args.seed,
        ))
    else:
        logger.warning("SASRec checkpoint not found at %s — skipping.", sasrec_path)

    # 4. Two-Tower + FAISS
    two_tower_path = args.artifacts_dir / "two_tower.pt"
    faiss_path = args.artifacts_dir / "faiss_items.index"
    if two_tower_path.exists() and faiss_path.exists():
        two_tower_model = load_two_tower(two_tower_path)
        faiss_index = FaissItemIndex.load(faiss_path, dim=two_tower_model.user_tower.mlp[-1].out_features)
        tt_fn = make_two_tower_recommend_fn(two_tower_model, faiss_index, train_sequences, args.max_len, args.eval_k)
        results.append(evaluate_recommender(
            "two_tower", tt_fn, train_sequences, val_sequences,
            args.eval_k, args.eval_users, args.seed,
        ))

        # 5. Cold-start check using the same Two-Tower + FAISS pipeline
        cold_metrics = evaluate_cold_start(
            tt_fn, cold_start_items, val_sequences, train_sequences,
            args.eval_k, args.eval_users, args.seed,
        )
        results.append(cold_metrics)
    else:
        logger.warning("Two-Tower checkpoint or FAISS index not found — skipping Two-Tower + cold-start eval.")

    # --- save results table ---
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("SEQUORA — BENCHMARK RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.artifacts_dir / "benchmark_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved benchmark results to %s", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SEQUORA full evaluation suite")
    parser.add_argument("--processed-dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--eval-users", type=int, default=2000)
    parser.add_argument("--als-factors", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run(parse_args())