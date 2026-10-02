"""
SEQUORA — Train Two-Tower
============================
Trains the Two-Tower retrieval model on real H&M sequences + the
multimodal content embeddings you already built, then builds a real
FAISS index from the trained item vectors.

Like train_sasrec.py, this subsamples users by default since training
runs on CPU — raise --max-train-users later once everything works.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.dataset import TwoTowerDataset, load_sequences
from data.preprocess import PreprocessConfig  # noqa: F401  (kept for reference/consistency)
from models.two_tower import TwoTower
from retrieval.faiss_index import FaissItemIndex

logger = logging.getLogger("sequora.train_two_tower")


def subsample_users(
    sequences: Dict[int, Dict[str, List]], max_users: int, seed: int
) -> Dict[int, Dict[str, List]]:
    if max_users is None or len(sequences) <= max_users:
        return sequences
    rng = random.Random(seed)
    keep_ids = rng.sample(list(sequences.keys()), max_users)
    subset = {uid: sequences[uid] for uid in keep_ids}
    logger.info("Subsampled users: %d -> %d", len(sequences), len(subset))
    return subset


def pad_left(seq: List[int], max_len: int) -> List[int]:
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


def load_content_lookup(embeddings_path: Path) -> Dict[int, np.ndarray]:
    """
    Loads item_content_embeddings.parquet (from models/multimodal.py) into
    {article_id(str) -> vector}. Articles not yet encoded (because you ran
    multimodal.py with --limit) simply won't be in this dict — those items
    fall back to a zero content vector during training, which is fine for
    getting the pipeline working end-to-end.
    """
    df = pd.read_parquet(embeddings_path)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    lookup = {
        row["article_id"]: row[dim_cols].to_numpy(dtype=np.float32)
        for _, row in df.iterrows()
    }
    logger.info("Loaded content vectors for %d articles (dim=%d)", len(lookup), len(dim_cols))
    return lookup


def build_content_matrix(
    article_id_map: Dict[str, int],
    content_lookup: Dict[str, np.ndarray],
    content_dim: int,
) -> np.ndarray:
    """
    (num_items + 1, content_dim) matrix indexed by item_idx, so content_vecs[item_idx]
    gives that item's content vector directly. Index 0 (padding) and any item
    without an encoded content vector get zeros.
    """
    num_items = max(article_id_map.values())
    matrix = np.zeros((num_items + 1, content_dim), dtype=np.float32)
    n_filled = 0
    for article_id, item_idx in article_id_map.items():
        vec = content_lookup.get(article_id)
        if vec is not None:
            matrix[item_idx] = vec
            n_filled += 1
    logger.info("Content matrix filled for %d / %d items", n_filled, num_items)
    return matrix


def collate_with_content(
    batch: List[Dict[str, torch.Tensor]],
    sequences: Dict[int, Dict[str, List]],
    content_matrix: np.ndarray,
    max_len: int,
) -> Dict[str, torch.Tensor]:
    """Builds the (input_seq, item_id, content_vec) triple each training
    step needs: input_seq is the user's history EXCLUDING the target item,
    so the model can't just memorize the answer from the sequence."""
    user_ids = [b["user_id"].item() for b in batch]
    item_ids = [b["item_id"].item() for b in batch]

    input_seqs = []
    for user_id, item_id in zip(user_ids, item_ids):
        items = sequences[user_id]["items"]
        # use everything up to (not including) this specific occurrence of item_id
        if item_id in items:
            cut = items.index(item_id)
        else:
            cut = len(items)
        input_seqs.append(pad_left(items[:cut], max_len))

    content_vecs = content_matrix[item_ids]

    return {
        "input_seq": torch.tensor(input_seqs, dtype=torch.long),
        "item_id": torch.tensor(item_ids, dtype=torch.long),
        "content_vec": torch.tensor(content_vecs, dtype=torch.float32),
    }


def train(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    logger.info("Using device: %s", device)

    article_id_map = json.loads((args.processed_dir / "article_id_map.json").read_text())
    num_items = max(article_id_map.values())

    train_sequences = load_sequences(args.processed_dir / "sequences_train.pkl")
    train_sequences = subsample_users(train_sequences, args.max_train_users, args.seed)

    content_lookup = load_content_lookup(args.content_embeddings_path)
    content_matrix = build_content_matrix(article_id_map, content_lookup, args.content_dim)

    dataset = TwoTowerDataset(train_sequences)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_with_content(
            batch, train_sequences, content_matrix, args.max_len
        ),
        drop_last=True,  # in-batch negatives need a full batch every step
    )
    logger.info("Training batches per epoch: %d", len(loader))

    model = TwoTower(
        num_items=num_items,
        content_dim=args.content_dim,
        item_emb_dim=args.item_emb_dim,
        output_dim=args.output_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0

        for step, batch in enumerate(loader, start=1):
            input_seq = batch["input_seq"].to(device)
            item_id = batch["item_id"].to(device)
            content_vec = batch["content_vec"].to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(input_seq, item_id, content_vec)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % args.log_every == 0:
                logger.info(
                    "epoch %d step %d/%d — running avg loss %.4f",
                    epoch, step, len(loader), total_loss / step,
                )

        avg_loss = total_loss / max(len(loader), 1)
        elapsed = time.time() - epoch_start
        logger.info("Epoch %d done in %.1fs — avg loss %.4f", epoch, elapsed, avg_loss)

    # --- save model checkpoint ---
    checkpoint_path = args.artifacts_dir / "two_tower.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_items": num_items,
            "content_dim": args.content_dim,
            "item_emb_dim": args.item_emb_dim,
            "output_dim": args.output_dim,
        },
        checkpoint_path,
    )
    logger.info("Saved Two-Tower checkpoint to %s", checkpoint_path)

    # --- build the real FAISS index from trained item vectors ---
    logger.info("Encoding all items for the FAISS index...")
    model.eval()
    all_item_ids = np.arange(1, num_items + 1)
    index = FaissItemIndex(dim=args.output_dim)

    batch_size = 2048
    with torch.no_grad():
        for start in range(0, len(all_item_ids), batch_size):
            chunk_ids = all_item_ids[start : start + batch_size]
            chunk_content = content_matrix[chunk_ids]
            id_tensor = torch.tensor(chunk_ids, dtype=torch.long)
            content_tensor = torch.tensor(chunk_content, dtype=torch.float32)
            vecs = model.encode_items(id_tensor, content_tensor).numpy()
            index.add(chunk_ids.astype(np.int64), vecs)

    index_path = args.artifacts_dir / "faiss_items.index"
    index.save(index_path)
    logger.info("Built and saved FAISS index with %d items to %s", index.size, index_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Two-Tower on SEQUORA data")
    parser.add_argument("--processed-dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--content-embeddings-path", type=Path,
                         default=Path("artifacts/item_content_embeddings.parquet"))
    parser.add_argument("--max-train-users", type=int, default=50000)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--content-dim", type=int, default=128)
    parser.add_argument("--item-emb-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(42)
    random.seed(42)
    train(parse_args())