"""
SEQUORA — Data Preprocessing
=============================
Cleans the H&M Personalized Fashion Recommendations raw tables, builds
customer/article ID mappings, performs a strictly temporal train/val/test
split, and constructs per-user chronological interaction sequences.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("sequora.preprocess")


@dataclass(frozen=True)
class PreprocessConfig:
    raw_dir: Path
    out_dir: Path
    val_cutoff: str = "2020-09-08"
    test_cutoff: str = "2020-09-15"
    min_user_interactions: int = 3
    max_sequence_len: int = 50
    seed: int = 42
    transactions_file: str = "transactions_train.csv"
    articles_file: str = "articles.csv"
    customers_file: str = "customers.csv"
    text_cols: Tuple[str, ...] = field(
        default_factory=lambda: ("product_type_name", "department_name", "detail_desc")
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_raw_data(cfg: PreprocessConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Loading raw CSVs from %s", cfg.raw_dir)

    transactions = pd.read_csv(
        cfg.raw_dir / cfg.transactions_file,
        dtype={"article_id": str, "customer_id": str},
        parse_dates=["t_dat"],
    )
    articles = pd.read_csv(cfg.raw_dir / cfg.articles_file, dtype={"article_id": str})
    customers = pd.read_csv(cfg.raw_dir / cfg.customers_file, dtype={"customer_id": str})

    logger.info(
        "Loaded: transactions=%d articles=%d customers=%d",
        len(transactions), len(articles), len(customers),
    )
    return transactions, articles, customers


def clean_articles(articles: pd.DataFrame, text_cols: Tuple[str, ...]) -> pd.DataFrame:
    df = articles.copy()

    for col in text_cols:
        if col not in df.columns:
            logger.warning("Expected text column '%s' missing — filling as empty", col)
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["text_blob"] = df[list(text_cols)].agg(" ".join, axis=1).str.strip()

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df["image_path"] = df["article_id"].apply(lambda a: f"images/0{a[:2]}/0{a}.jpg")
    df["has_image_placeholder"] = False

    n_missing_desc = (df["text_blob"].str.len() == 0).sum()
    if n_missing_desc:
        logger.warning("%d articles have completely empty text metadata", n_missing_desc)

    return df.drop_duplicates(subset="article_id").reset_index(drop=True)


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    df = customers.copy()
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["age"] = df["age"].fillna(df["age"].median())
    return df.drop_duplicates(subset="customer_id").reset_index(drop=True)


def clean_transactions(
    transactions: pd.DataFrame,
    valid_customers: set,
    valid_articles: set,
) -> pd.DataFrame:
    df = transactions.copy()
    before = len(df)

    df = df[df["customer_id"].isin(valid_customers) & df["article_id"].isin(valid_articles)]
    df = df.dropna(subset=["t_dat", "customer_id", "article_id"])
    df = df.sort_values("t_dat").reset_index(drop=True)

    logger.info("Transactions after cleaning: %d (dropped %d)", len(df), before - len(df))
    return df


def build_id_mappings(
    customers: pd.DataFrame, articles: pd.DataFrame
) -> Tuple[Dict[str, int], Dict[str, int]]:
    customer_id_map = {cid: idx for idx, cid in enumerate(customers["customer_id"].unique())}
    article_id_map = {aid: idx + 1 for idx, aid in enumerate(articles["article_id"].unique())}

    logger.info(
        "Built ID maps: %d customers, %d articles (+1 padding index)",
        len(customer_id_map), len(article_id_map),
    )
    return customer_id_map, article_id_map


def temporal_split(
    transactions: pd.DataFrame, val_cutoff: str, test_cutoff: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    val_cutoff_ts = pd.Timestamp(val_cutoff)
    test_cutoff_ts = pd.Timestamp(test_cutoff)

    train = transactions[transactions["t_dat"] < val_cutoff_ts]
    val = transactions[
        (transactions["t_dat"] >= val_cutoff_ts) & (transactions["t_dat"] < test_cutoff_ts)
    ]
    test = transactions[transactions["t_dat"] >= test_cutoff_ts]

    logger.info(
        "Temporal split -> train=%d (< %s), val=%d (%s to %s), test=%d (>= %s)",
        len(train), val_cutoff, len(val), val_cutoff, test_cutoff, len(test), test_cutoff,
    )

    if len(val) == 0 or len(test) == 0:
        logger.warning(
            "Val or test split is empty — check that val_cutoff/test_cutoff fall "
            "within the dataset's date range."
        )
    return train, val, test


def build_sequences(
    transactions: pd.DataFrame,
    customer_id_map: Dict[str, int],
    article_id_map: Dict[str, int],
    max_len: int,
) -> Dict[int, Dict[str, List]]:
    df = transactions.copy()
    df["user_idx"] = df["customer_id"].map(customer_id_map)
    df["item_idx"] = df["article_id"].map(article_id_map)
    df = df.dropna(subset=["user_idx", "item_idx"])
    df["user_idx"] = df["user_idx"].astype(int)
    df["item_idx"] = df["item_idx"].astype(int)
    df["unix_ts"] = df["t_dat"].astype("int64") // 10**9

    sequences: Dict[int, Dict[str, List]] = {}
    for user_idx, group in df.sort_values("t_dat").groupby("user_idx"):
        items = group["item_idx"].tolist()[-max_len:]
        ts = group["unix_ts"].tolist()[-max_len:]
        sequences[int(user_idx)] = {"items": items, "ts": ts}

    logger.info("Built sequences for %d users", len(sequences))
    return sequences


def filter_min_interactions(
    sequences: Dict[int, Dict[str, List]], min_interactions: int
) -> Dict[int, Dict[str, List]]:
    filtered = {u: s for u, s in sequences.items() if len(s["items"]) >= min_interactions}
    logger.info(
        "Filtered users with < %d interactions: %d -> %d",
        min_interactions, len(sequences), len(filtered),
    )
    return filtered


def find_cold_start_items(
    train_transactions: pd.DataFrame,
    article_id_map: Dict[str, int],
) -> List[int]:
    seen = set(train_transactions["article_id"].unique())
    cold = [idx for aid, idx in article_id_map.items() if aid not in seen]
    logger.info("Identified %d cold-start (zero train interaction) items", len(cold))
    return cold


def run(cfg: PreprocessConfig) -> None:
    set_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    transactions, articles, customers = load_raw_data(cfg)

    articles_clean = clean_articles(articles, cfg.text_cols)
    customers_clean = clean_customers(customers)
    transactions_clean = clean_transactions(
        transactions,
        valid_customers=set(customers_clean["customer_id"]),
        valid_articles=set(articles_clean["article_id"]),
    )

    customer_id_map, article_id_map = build_id_mappings(customers_clean, articles_clean)

    train_tx, val_tx, test_tx = temporal_split(transactions_clean, cfg.val_cutoff, cfg.test_cutoff)

    train_sequences = build_sequences(train_tx, customer_id_map, article_id_map, cfg.max_sequence_len)
    train_sequences = filter_min_interactions(train_sequences, cfg.min_user_interactions)

    val_sequences = build_sequences(val_tx, customer_id_map, article_id_map, cfg.max_sequence_len)
    test_sequences = build_sequences(test_tx, customer_id_map, article_id_map, cfg.max_sequence_len)

    cold_start_items = find_cold_start_items(train_tx, article_id_map)

    (cfg.out_dir / "customer_id_map.json").write_text(json.dumps(customer_id_map))
    (cfg.out_dir / "article_id_map.json").write_text(json.dumps(article_id_map))

    articles_clean.to_parquet(cfg.out_dir / "articles_clean.parquet", index=False)
    customers_clean.to_parquet(cfg.out_dir / "customers_clean.parquet", index=False)
    transactions_clean.to_parquet(cfg.out_dir / "transactions_clean.parquet", index=False)

    with open(cfg.out_dir / "sequences_train.pkl", "wb") as f:
        pickle.dump(train_sequences, f)
    with open(cfg.out_dir / "sequences_val.pkl", "wb") as f:
        pickle.dump(val_sequences, f)
    with open(cfg.out_dir / "sequences_test.pkl", "wb") as f:
        pickle.dump(test_sequences, f)

    (cfg.out_dir / "cold_start_items.json").write_text(json.dumps(cold_start_items))

    manifest = {
        "val_cutoff": cfg.val_cutoff,
        "test_cutoff": cfg.test_cutoff,
        "min_user_interactions": cfg.min_user_interactions,
        "max_sequence_len": cfg.max_sequence_len,
        "seed": cfg.seed,
        "n_customers": len(customer_id_map),
        "n_articles": len(article_id_map),
        "n_train_transactions": len(train_tx),
        "n_val_transactions": len(val_tx),
        "n_test_transactions": len(test_tx),
        "n_train_users": len(train_sequences),
        "n_cold_start_items": len(cold_start_items),
    }
    (cfg.out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Preprocessing complete. Manifest: %s", manifest)


def parse_args() -> PreprocessConfig:
    parser = argparse.ArgumentParser(description="SEQUORA data preprocessing")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-cutoff", type=str, default="2020-09-08")
    parser.add_argument("--test-cutoff", type=str, default="2020-09-15")
    parser.add_argument("--min-user-interactions", type=int, default=3)
    parser.add_argument("--max-sequence-len", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return PreprocessConfig(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        val_cutoff=args.val_cutoff,
        test_cutoff=args.test_cutoff,
        min_user_interactions=args.min_user_interactions,
        max_sequence_len=args.max_sequence_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run(parse_args())