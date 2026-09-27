"""
SEQUORA — PyTorch Datasets
============================
Turns the sequence files produced by data/preprocess.py into batches
that the SASRec and Two-Tower models can actually train on.
"""

from __future__ import annotations

import logging
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger("sequora.dataset")


def load_sequences(path: Path) -> Dict[int, Dict[str, List]]:
    with open(path, "rb") as f:
        sequences = pickle.load(f)
    logger.info("Loaded %d user sequences from %s", len(sequences), path)
    return sequences


def _pad_left(seq: List[int], max_len: int, pad_value: int = 0) -> List[int]:
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad_value] * (max_len - len(seq)) + seq


class SASRecDataset(Dataset):
    def __init__(
        self,
        sequences: Dict[int, Dict[str, List]],
        num_items: int,
        max_len: int = 50,
        seed: int = 42,
    ) -> None:
        self.user_ids = list(sequences.keys())
        self.sequences = sequences
        self.num_items = num_items
        self.max_len = max_len
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.user_ids)

    def _sample_negative(self, positive_set: set) -> int:
        while True:
            candidate = self.rng.randint(1, self.num_items)
            if candidate not in positive_set:
                return candidate

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        user_id = self.user_ids[idx]
        items = self.sequences[user_id]["items"]

        if len(items) < 2:
            input_seq = [0] * self.max_len
            target_seq = [0] * self.max_len
            neg_seq = [0] * self.max_len
        else:
            input_items = items[:-1]
            target_items = items[1:]
            positive_set = set(items)

            input_seq = _pad_left(input_items, self.max_len)
            target_seq = _pad_left(target_items, self.max_len)
            neg_seq = [
                self._sample_negative(positive_set) if t != 0 else 0
                for t in target_seq
            ]

        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "input_seq": torch.tensor(input_seq, dtype=torch.long),
            "target_seq": torch.tensor(target_seq, dtype=torch.long),
            "neg_seq": torch.tensor(neg_seq, dtype=torch.long),
        }


class TwoTowerDataset(Dataset):
    def __init__(self, sequences: Dict[int, Dict[str, List]]) -> None:
        self.pairs: List[Tuple[int, int]] = []
        for user_id, data in sequences.items():
            for item_id in data["items"]:
                self.pairs.append((user_id, item_id))
        logger.info("Built %d (user, item) pairs for Two-Tower training", len(self.pairs))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        user_id, item_id = self.pairs[idx]
        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "item_id": torch.tensor(item_id, dtype=torch.long),
        }


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("processed_data"))
    args = parser.parse_args()

    article_id_map = json.loads((args.processed_dir / "article_id_map.json").read_text())
    num_items = max(article_id_map.values())

    train_sequences = load_sequences(args.processed_dir / "sequences_train.pkl")

    sasrec_ds = SASRecDataset(train_sequences, num_items=num_items, max_len=50)
    two_tower_ds = TwoTowerDataset(train_sequences)

    print(f"SASRecDataset size: {len(sasrec_ds)}")
    print(f"TwoTowerDataset size: {len(two_tower_ds)}")

    sample = sasrec_ds[0]
    print("Sample SASRec item shapes:")
    for k, v in sample.items():
        print(f"  {k}: {v.shape}")