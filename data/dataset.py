"""
SEQUORA — Datasets (fast version)
====================================
Turns the sequence files produced by data/preprocess.py into training
batches for SASRec and Two-Tower.

Why this was rewritten: the old datasets built every sample one at a time
in Python (one random negative per position, an items.index() lookup per
Two-Tower sample), so the data loader — not the model — was the bottleneck,
and a GPU could not fix that. Now every user's padded sequence is built
ONCE as a tensor, and batches are cut straight out of it on the device.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch

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


def build_padded_matrix(
    sequences: Dict[int, Dict[str, List]], width: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (user_ids, matrix, lens). Each matrix row holds that user's most
    recent `width` items, left-padded with 0 (item index 0 is reserved for padding)."""
    n = len(sequences)
    user_ids = np.zeros(n, dtype=np.int64)
    matrix = np.zeros((n, width), dtype=np.int32)
    lens = np.zeros(n, dtype=np.int32)
    for row, (uid, data) in enumerate(sequences.items()):
        items = data["items"][-width:]
        k = len(items)
        user_ids[row] = uid
        lens[row] = k
        if k:
            matrix[row, width - k:] = items
    return user_ids, matrix, lens


def sample_negatives(input_seq: torch.Tensor, target_seq: torch.Tensor, num_items: int) -> torch.Tensor:
    """One random negative per position, drawn on the same device as the batch.
    Anything that collides with the user's own items is re-drawn once (a leftover
    collision has probability ~0.05%). Padded positions get 0."""
    neg = torch.randint(1, num_items + 1, target_seq.shape, device=target_seq.device)
    own_items = torch.cat([input_seq, target_seq[:, -1:]], dim=1)          # (B, L+1)
    clash = (neg.unsqueeze(-1) == own_items.unsqueeze(1)).any(dim=-1)      # (B, L)
    redraw = torch.randint(1, num_items + 1, target_seq.shape, device=target_seq.device)
    neg = torch.where(clash, redraw, neg)
    return neg * (target_seq != 0)


class SASRecDataset:
    """All (input, target) sequences pre-built as tensors. Same pairing as the
    original: input = items[:-1], target = items[1:], both left-padded to max_len."""

    def __init__(
        self,
        sequences: Dict[int, Dict[str, List]],
        num_items: int,
        max_len: int = 50,
        seed: int = 42,
        device: torch.device | str = "cpu",
    ) -> None:
        _, full, lens = build_padded_matrix(sequences, max_len + 1)
        full = full[lens >= 2]                      # need at least one (input, target) pair
        inp = full[:, :-1]
        tgt = full[:, 1:].copy()
        tgt[inp == 0] = 0                           # no target where the input is padding

        self.num_items = num_items
        self.max_len = max_len
        self.device = torch.device(device)
        self.input = torch.from_numpy(np.ascontiguousarray(inp)).to(self.device)
        self.target = torch.from_numpy(np.ascontiguousarray(tgt)).to(self.device)
        logger.info("SASRecDataset: %d users, tensors on %s", len(self), self.device)

    def __len__(self) -> int:
        return self.input.shape[0]

    def num_batches(self, batch_size: int) -> int:
        return (len(self) + batch_size - 1) // batch_size

    def batches(self, batch_size: int, shuffle: bool = True) -> Iterator[Dict[str, torch.Tensor]]:
        n = len(self)
        order = torch.randperm(n, device=self.device) if shuffle else torch.arange(n, device=self.device)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            input_seq = self.input[idx].long()
            target_seq = self.target[idx].long()
            yield {
                "input_seq": input_seq,
                "target_seq": target_seq,
                "neg_seq": sample_negatives(input_seq, target_seq, self.num_items),
            }


class TwoTowerDataset:
    """Each draw = a random user, a random position in their history, the
    history BEFORE that position as input and the item AT that position as the
    target. Everything happens on the device; there is no per-sample Python."""

    def __init__(
        self,
        sequences: Dict[int, Dict[str, List]],
        max_len: int = 50,
        device: torch.device | str = "cpu",
    ) -> None:
        _, matrix, lens = build_padded_matrix(sequences, max_len)
        keep = lens >= 2
        self.max_len = max_len
        self.device = torch.device(device)
        self.seq = torch.from_numpy(matrix[keep]).to(self.device)
        self.lens = torch.from_numpy(lens[keep]).to(self.device)
        logger.info("TwoTowerDataset: %d users, tensors on %s", len(self), self.device)

    def __len__(self) -> int:
        return self.seq.shape[0]

    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        L = self.max_len
        dev = self.device
        r = torch.randint(0, len(self), (batch_size,), device=dev)
        rows = self.seq[r].long()                                   # (B, L), left-padded
        lens = self.lens[r].long()                                  # (B,)

        # target column p in [L-len+1, L-1] so at least one history item precedes it
        offset = (torch.rand(batch_size, device=dev) * (lens - 1)).long()
        offset = torch.minimum(offset, lens - 2)
        p = (L - lens + 1) + offset

        item_id = rows.gather(1, p.unsqueeze(1)).squeeze(1)

        # history = rows[:, :p], shifted right so it is left-padded again
        idx = torch.arange(L, device=dev).unsqueeze(0) - (L - p).unsqueeze(1)
        input_seq = rows.gather(1, idx.clamp(min=0)) * (idx >= 0).long()
        return {"input_seq": input_seq, "item_id": item_id}


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
    two_tower_ds = TwoTowerDataset(train_sequences, max_len=50)

    print(f"SASRecDataset users: {len(sasrec_ds)}")
    print(f"TwoTowerDataset users: {len(two_tower_ds)}")

    batch = next(sasrec_ds.batches(4))
    print("Sample SASRec batch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)}")
    batch = two_tower_ds.sample_batch(4)
    print("Sample Two-Tower batch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)}")