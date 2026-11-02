"""
SEQUORA — Train Two-Tower
============================
Trains the Two-Tower retrieval model on real H&M sequences + the
multimodal content embeddings you already built, then builds a real
FAISS index from the trained item vectors.

v2 changes:
  * uses the GPU automatically when one is available (CPU otherwise)
  * batches are drawn directly on the device (data/dataset.py) — no per-sample
    Python loops and no items.index() lookups
  * an "epoch" is now --steps-per-epoch random batches (the old epoch was all
    31M interactions, far too long to be useful)
  * a missing content-embeddings file is allowed (content vectors become zeros)
  * --max-train-users 0 means "use everyone"
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

from data.dataset import TwoTowerDataset, load_sequences
from models.two_tower import TwoTower
from retrieval.faiss_index import FaissItemIndex

logger = logging.getLogger("sequora.train_two_tower")


def subsample_users(
    sequences: Dict[int, Dict[str, List]], max_users: int, seed: int
) -> Dict[int, Dict[str, List]]:
    if not max_users or len(sequences) <= max_users:
        return sequences
    rng = random.Random(seed)
    keep_ids = rng.sample(list(sequences.keys()), max_users)
    subset = {uid: sequences[uid] for uid in keep_ids}
    logger.info("Subsampled users: %d -> %d", len(sequences), len(subset))
    return subset


def load_content_lookup(embeddings_path: Path) -> Dict[str, np.ndarray]:
    """
    Loads item_content_embeddings.parquet (from models/multimodal.py) into
    {article_id(str) -> vector}. Articles not yet encoded simply won't be in
    this dict — those items get a zero content vector during training.
    """
    if not embeddings_path.exists():
        logger.warning("No content embeddings at %s — using zero content vectors.", embeddings_path)
        return {}
    df = pd.read_parquet(embeddings_path)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    vectors = df[dim_cols].to_numpy(dtype=np.float32)
    ids = df["article_id"].astype(str).str.zfill(10).tolist()
    logger.info("Loaded content vectors for %d articles (dim=%d)", len(ids), len(dim_cols))
    return dict(zip(ids, vectors))


def build_content_matrix(
    article_id_map: Dict[str, int],
    content_lookup: Dict[str, np.ndarray],
    content_dim: int,
) -> np.ndarray:
    """(num_items + 1, content_dim) matrix indexed by item_idx. Index 0 (padding)
    and any item without an encoded content vector get zeros."""
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


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    article_id_map = json.loads((args.processed_dir / "article_id_map.json").read_text())
    num_items = max(article_id_map.values())

    train_sequences = load_sequences(args.processed_dir / "sequences_train.pkl")
    train_sequences = subsample_users(train_sequences, args.max_train_users, args.seed)

    content_lookup = load_content_lookup(args.content_embeddings_path)
    content_matrix = build_content_matrix(article_id_map, content_lookup, args.content_dim)
    content_gpu = torch.from_numpy(content_matrix).to(device)

    t0 = time.time()
    dataset = TwoTowerDataset(train_sequences, max_len=args.max_len, device=device)
    logger.info("Built training tensors in %.1fs", time.time() - t0)

    model = TwoTower(
        num_items=num_items,
        content_dim=args.content_dim,
        item_emb_dim=args.item_emb_dim,
        output_dim=args.output_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(path: Path) -> None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_items": num_items,
                "content_dim": args.content_dim,
                "item_emb_dim": args.item_emb_dim,
                "output_dim": args.output_dim,
            },
            path,
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0

        for step in range(1, args.steps_per_epoch + 1):
            batch = dataset.sample_batch(args.batch_size)
            input_seq = batch["input_seq"]
            item_id = batch["item_id"]
            content_vec = content_gpu[item_id]

            optimizer.zero_grad()
            loss = model.compute_loss(input_seq, item_id, content_vec)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % args.log_every == 0:
                logger.info(
                    "epoch %d step %d/%d — running avg loss %.4f",
                    epoch, step, args.steps_per_epoch, total_loss / step,
                )

        avg_loss = total_loss / max(args.steps_per_epoch, 1)
        elapsed = time.time() - epoch_start
        msg = f"Epoch {epoch} done in {elapsed:.1f}s — avg loss {avg_loss:.4f}"
        if device.type == "cuda":
            msg += f" — peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        logger.info(msg)
        save_checkpoint(args.artifacts_dir / f"two_tower_epoch{epoch}.pt")

    # --- save final model checkpoint ---
    checkpoint_path = args.artifacts_dir / "two_tower.pt"
    save_checkpoint(checkpoint_path)
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
            id_tensor = torch.tensor(chunk_ids, dtype=torch.long, device=device)
            content_tensor = content_gpu[id_tensor]
            vecs = model.encode_items(id_tensor, content_tensor).cpu().numpy()
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
    parser.add_argument("--max-train-users", type=int, default=100000,
                         help="Train on this many users. 0 = use everyone.")
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--content-dim", type=int, default=128)
    parser.add_argument("--item-emb-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=3000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(42)
    random.seed(42)
    train(parse_args())