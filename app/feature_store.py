"""
SEQUORA — Shared Feature Store
================================
A single source of truth for every engineered feature used by the LightGBM
re-ranker. The same methods are called offline (training) and online
(FastAPI serving), so definitions never drift between the two.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger("sequora.feature_store")

SECONDS_PER_DAY = 86_400


class FeatureContext(BaseModel):
    user_idx: int
    item_idx: int
    category: Optional[str] = None
    price: Optional[float] = Field(default=None, ge=0)
    current_ts: int


@dataclass
class FeatureAggregates:
    reference_ts: int
    user_last_category_ts: Dict[str, int]
    user_category_counts: Dict[str, int]
    user_total_counts: Dict[int, int]
    user_avg_price: Dict[int, float]
    item_interaction_counts: Dict[int, int]
    item_last_interaction_ts: Dict[int, int]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "reference_ts": self.reference_ts,
            "user_last_category_ts": self.user_last_category_ts,
            "user_category_counts": self.user_category_counts,
            "user_total_counts": self.user_total_counts,
            "user_avg_price": self.user_avg_price,
            "item_interaction_counts": self.item_interaction_counts,
            "item_last_interaction_ts": self.item_last_interaction_ts,
        }))
        logger.info("Saved feature aggregates to %s", path)

    @classmethod
    def load(cls, path: Path) -> "FeatureAggregates":
        raw = json.loads(path.read_text())
        return cls(
            reference_ts=raw["reference_ts"],
            user_last_category_ts=raw["user_last_category_ts"],
            user_category_counts=raw["user_category_counts"],
            user_total_counts={int(k): v for k, v in raw["user_total_counts"].items()},
            user_avg_price={int(k): v for k, v in raw["user_avg_price"].items()},
            item_interaction_counts={int(k): v for k, v in raw["item_interaction_counts"].items()},
            item_last_interaction_ts={int(k): v for k, v in raw["item_last_interaction_ts"].items()},
        )

    @classmethod
    def build_from_transactions(
        cls,
        transactions: pd.DataFrame,
        articles: pd.DataFrame,
        reference_ts: int,
        category_col: str = "product_type_name",
    ) -> "FeatureAggregates":
        df = transactions.copy()
        if "unix_ts" not in df.columns:
            df["unix_ts"] = df["t_dat"].astype("int64") // 10**9
        df = df[df["unix_ts"] <= reference_ts]

        df = df.merge(
            articles[["article_id", category_col, "price"] if "price" in articles.columns
                     else ["article_id", category_col]],
            on="article_id", how="left",
        )

        user_last_category_ts: Dict[str, int] = {}
        user_category_counts: Dict[str, int] = {}
        for (user_idx, category), group in df.groupby(["user_idx", category_col]):
            key = f"{user_idx}:{category}"
            user_last_category_ts[key] = int(group["unix_ts"].max())
            user_category_counts[key] = int(len(group))

        user_total_counts = df.groupby("user_idx").size().astype(int).to_dict()

        user_avg_price: Dict[int, float] = {}
        if "price" in df.columns:
            user_avg_price = df.groupby("user_idx")["price"].mean().fillna(0.0).to_dict()
            user_avg_price = {int(k): float(v) for k, v in user_avg_price.items()}

        item_interaction_counts = df.groupby("item_idx").size().astype(int).to_dict()
        item_last_interaction_ts = df.groupby("item_idx")["unix_ts"].max().astype(int).to_dict()

        return cls(
            reference_ts=reference_ts,
            user_last_category_ts=user_last_category_ts,
            user_category_counts=user_category_counts,
            user_total_counts={int(k): int(v) for k, v in user_total_counts.items()},
            user_avg_price=user_avg_price,
            item_interaction_counts={int(k): int(v) for k, v in item_interaction_counts.items()},
            item_last_interaction_ts={int(k): int(v) for k, v in item_last_interaction_ts.items()},
        )


class FeatureStore:
    def __init__(
        self,
        aggregates: FeatureAggregates,
        recency_half_life_days: float = 14.0,
        popularity_half_life_days: float = 7.0,
    ) -> None:
        self.agg = aggregates
        self.recency_half_life_seconds = recency_half_life_days * SECONDS_PER_DAY
        self.popularity_half_life_seconds = popularity_half_life_days * SECONDS_PER_DAY

    def recency_decay(self, user_idx: int, category: Optional[str], current_ts: int) -> float:
        if category is None:
            return 0.0
        key = f"{user_idx}:{category}"
        last_ts = self.agg.user_last_category_ts.get(key)
        if last_ts is None:
            return 0.0
        elapsed = max(0, current_ts - last_ts)
        return math.exp(-elapsed / self.recency_half_life_seconds)

    def user_category_affinity(self, user_idx: int, category: Optional[str]) -> float:
        if category is None:
            return 0.0
        key = f"{user_idx}:{category}"
        cat_count = self.agg.user_category_counts.get(key, 0)
        total = self.agg.user_total_counts.get(user_idx, 0)
        if total == 0:
            return 0.0
        return cat_count / total

    def price_affinity(self, user_idx: int, price: Optional[float]) -> float:
        if price is None:
            return 0.5
        avg = self.agg.user_avg_price.get(user_idx)
        if not avg or avg <= 0:
            return 0.5
        ratio = price / avg
        return float(math.exp(-abs(math.log(ratio))))

    def item_popularity_decay(self, item_idx: int, current_ts: int) -> float:
        count = self.agg.item_interaction_counts.get(item_idx, 0)
        last_ts = self.agg.item_last_interaction_ts.get(item_idx)
        if count == 0 or last_ts is None:
            return 0.0
        elapsed = max(0, current_ts - last_ts)
        decay = math.exp(-elapsed / self.popularity_half_life_seconds)
        return math.log1p(count) * decay

    def compute_online_features(self, ctx: FeatureContext) -> Dict[str, float]:
        return {
            "recency_decay": self.recency_decay(ctx.user_idx, ctx.category, ctx.current_ts),
            "user_category_affinity": self.user_category_affinity(ctx.user_idx, ctx.category),
            "price_affinity": self.price_affinity(ctx.user_idx, ctx.price),
            "item_popularity_decay": self.item_popularity_decay(ctx.item_idx, ctx.current_ts),
        }

    def compute_offline_features(
        self,
        candidates: pd.DataFrame,
        current_ts_col: str = "unix_ts",
        category_col: str = "product_type_name",
        price_col: str = "price",
    ) -> pd.DataFrame:
        records = []
        for row in candidates.itertuples(index=False):
            ctx = FeatureContext(
                user_idx=int(getattr(row, "user_idx")),
                item_idx=int(getattr(row, "item_idx")),
                category=getattr(row, category_col, None),
                price=getattr(row, price_col, None),
                current_ts=int(getattr(row, current_ts_col)),
            )
            records.append(self.compute_online_features(ctx))
        feature_df = pd.DataFrame.from_records(records)
        return pd.concat([candidates.reset_index(drop=True), feature_df], axis=1)


def load_feature_store(aggregates_path: Path, **kwargs) -> FeatureStore:
    aggregates = FeatureAggregates.load(aggregates_path)
    logger.info("Loaded feature aggregates (reference_ts=%d) from %s", aggregates.reference_ts, aggregates_path)
    return FeatureStore(aggregates=aggregates, **kwargs)