"""
SEQUORA — LightGBM Re-Ranker
===============================
Takes the top-N candidates retrieved by SASRec/Two-Tower/content cold-start
and re-scores them using engineered features (from app/feature_store.py)
plus each candidate's retrieval signals, so the final order reflects more
than raw similarity alone.

Uses LGBMRanker with the lambdarank objective — a learning-to-rank model,
not a plain classifier, since the goal is getting the ORDER right within
each user's candidate list, not predicting an isolated click probability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger("sequora.ranker")

FEATURE_COLUMNS = [
    "recency_decay",
    "user_category_affinity",
    "price_affinity",
    "item_popularity_decay",
    "sasrec_score",
    "two_tower_score",
    "source_sasrec",       # 1 if this candidate came from the sequential engine
    "source_cold_start",   # 1 if this candidate came from the content engine
    "source_retrieval",    # 1 if this candidate came from Two-Tower + FAISS
]


class Ranker:
    def __init__(self, **lgb_params) -> None:
        default_params = dict(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=5,
            random_state=42,
            verbosity=-1,
        )
        default_params.update(lgb_params)
        self.model = lgb.LGBMRanker(**default_params)
        self.is_fitted = False

    def fit(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        group_sizes: List[int],
    ) -> None:
        """
        features: (n_rows, n_features) — one row per (user, candidate) pair,
                  columns must match FEATURE_COLUMNS
        labels:   (n_rows,) relevance labels, e.g. 1 if the user actually
                  interacted with this candidate, 0 otherwise
        group_sizes: how many consecutive rows belong to each user's
                     candidate list — e.g. [100, 100, 87, ...] if the first
                     100 rows are user A's candidates, next 100 are user B's,
                     etc. Rows for the SAME user must be contiguous.
        """
        X = features[FEATURE_COLUMNS].to_numpy()
        self.model.fit(X, labels, group=group_sizes)
        self.is_fitted = True
        logger.info(
            "Trained ranker on %d rows across %d groups (avg group size %.1f)",
            len(features), len(group_sizes), len(features) / max(len(group_sizes), 1),
        )

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Higher score = should rank higher. Does NOT sort — caller decides
        how to use the scores (e.g. sort candidates per user)."""
        if not self.is_fitted:
            raise RuntimeError("Ranker has not been fitted yet — call fit() or load() first.")
        X = features[FEATURE_COLUMNS].to_numpy()
        return self.model.predict(X)

    def rerank(
        self, candidates: pd.DataFrame, top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Convenience wrapper: scores `candidates` (must have FEATURE_COLUMNS
        plus whatever identifying columns you want to keep, e.g. item_id),
        adds a 'rerank_score' column, and returns them sorted best-first.
        """
        result = candidates.copy()
        result["rerank_score"] = self.score(candidates)
        result = result.sort_values("rerank_score", ascending=False).reset_index(drop=True)
        if top_k is not None:
            result = result.head(top_k)
        return result

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)
        logger.info("Saved ranker to %s", path)

    @classmethod
    def load(cls, path: Path) -> "Ranker":
        wrapper = cls()
        wrapper.model = joblib.load(path)
        wrapper.is_fitted = True
        logger.info("Loaded ranker from %s", path)
        return wrapper


def _make_synthetic_training_data(
    n_users: int, candidates_per_user: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray, List[int]]:
    """
    Builds a fake but STRUCTURED dataset: within each user's candidate
    list, exactly one candidate has deliberately strong features (the
    'true positive'), the rest are noise. If training works, the model
    should learn to rank that one candidate at or near the top.
    """
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    group_sizes = []

    for _ in range(n_users):
        positive_idx = rng.integers(0, candidates_per_user)
        for i in range(candidates_per_user):
            is_positive = i == positive_idx
            rows.append({
                "recency_decay": rng.uniform(0.8, 1.0) if is_positive else rng.uniform(0.0, 0.3),
                "user_category_affinity": rng.uniform(0.7, 1.0) if is_positive else rng.uniform(0.0, 0.3),
                "price_affinity": rng.uniform(0.8, 1.0) if is_positive else rng.uniform(0.2, 0.6),
                "item_popularity_decay": rng.uniform(0.5, 1.0) if is_positive else rng.uniform(0.0, 0.5),
                "sasrec_score": rng.uniform(0.8, 1.0) if is_positive else rng.uniform(-0.2, 0.3),
                "two_tower_score": rng.uniform(0.8, 1.0) if is_positive else rng.uniform(-0.2, 0.3),
                "source_sasrec": 1 if is_positive else rng.integers(0, 2),
                "source_cold_start": 0,
                "source_retrieval": 1,
            })
            labels.append(1 if is_positive else 0)
        group_sizes.append(candidates_per_user)

    features = pd.DataFrame(rows)[FEATURE_COLUMNS]
    return features, np.array(labels), group_sizes


def _smoke_test() -> None:
    n_users = 200
    candidates_per_user = 20

    train_features, train_labels, train_groups = _make_synthetic_training_data(
        n_users=n_users, candidates_per_user=candidates_per_user, seed=42
    )
    ranker = Ranker()
    ranker.fit(train_features, train_labels, train_groups)
    print(f"[1/3] Training OK. Fitted on {len(train_features)} rows across {len(train_groups)} groups.")

    # evaluate: for held-out synthetic users, does the true positive rank #1?
    test_features, test_labels, test_groups = _make_synthetic_training_data(
        n_users=50, candidates_per_user=candidates_per_user, seed=99
    )
    scores = ranker.score(test_features)

    hits_at_1 = 0
    offset = 0
    for group_size in test_groups:
        group_scores = scores[offset : offset + group_size]
        group_labels = test_labels[offset : offset + group_size]
        top_idx = int(np.argmax(group_scores))
        if group_labels[top_idx] == 1:
            hits_at_1 += 1
        offset += group_size

    accuracy_at_1 = hits_at_1 / len(test_groups)
    print(f"[2/3] Learned ranking signal. True positive ranked #1 for {accuracy_at_1:.0%} of test users "
          f"(should be well above the {1/candidates_per_user:.0%} random baseline).")
    assert accuracy_at_1 > (1 / candidates_per_user) * 3, (
        "Ranker does not appear to have learned anything useful — accuracy barely above random."
    )

    # save/load round-trip
    tmp_path = Path("artifacts/_ranker_smoke_test.pkl")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    ranker.save(tmp_path)
    reloaded = Ranker.load(tmp_path)
    reloaded_scores = reloaded.score(test_features)
    assert np.allclose(scores, reloaded_scores), "Reloaded ranker gives different scores"
    tmp_path.unlink()
    print("[3/3] Save/load OK. Reloaded ranker matches the original exactly.")

    print("RANKER SMOKE TEST PASSED")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _smoke_test()