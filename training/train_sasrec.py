"""
SEQUORA — Train SASRec
========================
Trains the SASRec model on the processed H&M sequences and evaluates
next-item prediction with Recall@10 / NDCG@10 on the validation split.

v2 changes:
  * uses the GPU automatically when one is available (CPU otherwise)
  * fast tensor batches (data/dataset.py) instead of a per-sample Python loader
  * evaluation uses ALL train histories as context, so even a pilot run trained
    on a user subset is scored on thousands of validation users
  * saves a checkpoint after EVERY epoch (sasrec_epoch1.pt, ...) so you can
    compare 2 / 5 / 10 epochs from a single run, and keeps hit counts
  * --max-train-users 0 means "use everyone"
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from data.dataset import SASRecDataset, load_sequences
from models.sasrec import SASRec

logger = logging.getLogger("sequora.train_sasrec")


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


def pad_left(seq: List[int], max_len: int) -> List[int]:
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


@torch.no_grad()
def evaluate(
    model: SASRec,
    context_sequences: Dict[int, Dict[str, List]],
    val_sequences: Dict[int, Dict[str, List]],
    max_len: int,
    k: int,
    max_eval_users: int,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Next-item evaluation: context = the user's train history, target = the
    first item they interacted with in the validation period. A hit means
    the true next item appeared in the model's top-k recommendations.
    """
    model.eval()

    eligible = sorted(uid for uid in val_sequences if uid in context_sequences)
    if not eligible:
        logger.warning("No users overlap between train and val sequences — eval skipped.")
        return {"recall_at_k": 0.0, "ndcg_at_k": 0.0, "hits": 0, "n_eval_users": 0}

    rng = random.Random(seed)
    if len(eligible) > max_eval_users:
        eligible = rng.sample(eligible, max_eval_users)

    hits = 0
    ndcg_sum = 0.0
    batch_size = 256

    for start in range(0, len(eligible), batch_size):
        batch_ids = eligible[start : start + batch_size]
        inputs = [pad_left(context_sequences[uid]["items"], max_len) for uid in batch_ids]
        targets = [val_sequences[uid]["items"][0] for uid in batch_ids]

        input_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
        top_k = model.recommend(input_tensor, k=k, exclude_seen=True)

        for row, target in enumerate(targets):
            ranked = top_k[row].tolist()
            if target in ranked:
                hits += 1
                rank = ranked.index(target) + 1
                ndcg_sum += 1.0 / float(np.log2(rank + 1.0))

    n = len(eligible)
    return {
        "recall_at_k": hits / n,
        "ndcg_at_k": ndcg_sum / n,
        "hits": hits,
        "n_eval_users": n,
    }


def save_checkpoint(model: SASRec, path: Path, num_items: int, args: argparse.Namespace) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_items": num_items,
            "max_len": args.max_len,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_blocks": args.num_blocks,
            "dropout": args.dropout,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    article_id_map = json.loads((args.processed_dir / "article_id_map.json").read_text())
    num_items = max(article_id_map.values())
    logger.info("num_items (catalog size) = %d", num_items)

    all_train_sequences = load_sequences(args.processed_dir / "sequences_train.pkl")
    val_sequences = load_sequences(args.processed_dir / "sequences_val.pkl")

    train_sequences = subsample_users(all_train_sequences, args.max_train_users, args.seed)

    t0 = time.time()
    dataset = SASRecDataset(
        train_sequences, num_items=num_items, max_len=args.max_len, seed=args.seed, device=device
    )
    n_batches = dataset.num_batches(args.batch_size)
    logger.info("Built training tensors in %.1fs — %d batches per epoch", time.time() - t0, n_batches)

    model = SASRec(
        num_items=num_items,
        max_len=args.max_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0

        for step, batch in enumerate(dataset.batches(args.batch_size, shuffle=True), start=1):
            optimizer.zero_grad()
            loss = model.compute_loss(batch["input_seq"], batch["target_seq"], batch["neg_seq"])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % args.log_every == 0:
                logger.info(
                    "epoch %d step %d/%d — running avg loss %.4f",
                    epoch, step, n_batches, total_loss / step,
                )

        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - epoch_start
        msg = f"Epoch {epoch} done in {elapsed:.1f}s — avg loss {avg_loss:.4f}"
        if device.type == "cuda":
            msg += f" — peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        logger.info(msg)

        metrics = evaluate(
            model, all_train_sequences, val_sequences,
            max_len=args.max_len, k=args.eval_k,
            max_eval_users=args.eval_users, seed=args.seed, device=device,
        )
        logger.info(
            "Epoch %d eval — Recall@%d = %.4f, NDCG@%d = %.4f (hits=%d of n=%d users)",
            epoch, args.eval_k, metrics["recall_at_k"],
            args.eval_k, metrics["ndcg_at_k"], metrics["hits"], metrics["n_eval_users"],
        )
        history.append({"epoch": epoch, "avg_loss": avg_loss, "seconds": elapsed, **metrics})

        save_checkpoint(model, args.artifacts_dir / f"sasrec_epoch{epoch}.pt", num_items, args)

        (args.artifacts_dir / "sasrec_metrics.json").write_text(
            json.dumps({"final": history[-1], "history": history}, indent=2)
        )

    final_path = args.artifacts_dir / "sasrec.pt"
    shutil.copyfile(args.artifacts_dir / f"sasrec_epoch{args.epochs}.pt", final_path)
    logger.info("Saved final model checkpoint to %s", final_path)
    logger.info("Final metrics: %s", history[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SASRec on SEQUORA data")
    parser.add_argument("--processed-dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-train-users", type=int, default=100000,
                         help="Train on this many users. 0 = use everyone.")
    parser.add_argument("--eval-users", type=int, default=10000,
                         help="Validation users to score after each epoch.")
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(42)
    random.seed(42)
    train(parse_args())