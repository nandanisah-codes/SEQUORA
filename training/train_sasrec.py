"""
SEQUORA — Train SASRec
========================
Trains the SASRec model on the processed H&M sequences and evaluates
next-item prediction with Recall@10 / NDCG@10 on the validation split.

Because this typically runs on CPU, --max-train-users lets you train on a
slice of users first to get a real, working result quickly. Remove that
flag (or raise it) later to train on everything, ideally on a machine
with a GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from data.dataset import SASRecDataset, load_sequences
from models.sasrec import SASRec

logger = logging.getLogger("sequora.train_sasrec")


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


@torch.no_grad()
def evaluate(
    model: SASRec,
    train_sequences: Dict[int, Dict[str, List]],
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

    eligible = [uid for uid in val_sequences if uid in train_sequences]
    if not eligible:
        logger.warning("No users overlap between train and val sequences — eval skipped.")
        return {"recall_at_k": 0.0, "ndcg_at_k": 0.0, "n_eval_users": 0}

    rng = random.Random(seed)
    if len(eligible) > max_eval_users:
        eligible = rng.sample(eligible, max_eval_users)

    hits = 0
    ndcg_sum = 0.0
    batch_size = 256

    for start in range(0, len(eligible), batch_size):
        batch_ids = eligible[start : start + batch_size]
        inputs = [pad_left(train_sequences[uid]["items"], max_len) for uid in batch_ids]
        targets = [val_sequences[uid]["items"][0] for uid in batch_ids]

        input_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
        top_k = model.recommend(input_tensor, k=k, exclude_seen=True)

        for row, target in enumerate(targets):
            ranked = top_k[row].tolist()
            if target in ranked:
                hits += 1
                rank = ranked.index(target) + 1
                ndcg_sum += 1.0 / torch.log2(torch.tensor(rank + 1.0)).item()

    n = len(eligible)
    return {
        "recall_at_k": hits / n,
        "ndcg_at_k": ndcg_sum / n,
        "n_eval_users": n,
    }


def train(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    logger.info("Using device: %s", device)

    article_id_map = json.loads((args.processed_dir / "article_id_map.json").read_text())
    num_items = max(article_id_map.values())
    logger.info("num_items (catalog size) = %d", num_items)

    train_sequences = load_sequences(args.processed_dir / "sequences_train.pkl")
    val_sequences = load_sequences(args.processed_dir / "sequences_val.pkl")

    train_sequences = subsample_users(train_sequences, args.max_train_users, args.seed)

    dataset = SASRecDataset(
        train_sequences, num_items=num_items, max_len=args.max_len, seed=args.seed
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    logger.info("Training batches per epoch: %d", len(loader))

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

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0

        for step, batch in enumerate(loader, start=1):
            input_seq = batch["input_seq"].to(device)
            target_seq = batch["target_seq"].to(device)
            neg_seq = batch["neg_seq"].to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(input_seq, target_seq, neg_seq)
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

        metrics = evaluate(
            model, train_sequences, val_sequences,
            max_len=args.max_len, k=args.eval_k,
            max_eval_users=args.eval_users, seed=args.seed, device=device,
        )
        logger.info(
            "Epoch %d eval — Recall@%d = %.4f, NDCG@%d = %.4f (n=%d users)",
            epoch, args.eval_k, metrics["recall_at_k"],
            args.eval_k, metrics["ndcg_at_k"], metrics["n_eval_users"],
        )

    checkpoint_path = args.artifacts_dir / "sasrec.pt"
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
        checkpoint_path,
    )
    logger.info("Saved model checkpoint to %s", checkpoint_path)

    final_metrics = evaluate(
        model, train_sequences, val_sequences,
        max_len=args.max_len, k=args.eval_k,
        max_eval_users=args.eval_users, seed=args.seed, device=device,
    )
    metrics_path = args.artifacts_dir / "sasrec_metrics.json"
    metrics_path.write_text(json.dumps(final_metrics, indent=2))
    logger.info("Final metrics: %s", final_metrics)
    logger.info("Saved metrics to %s", metrics_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SASRec on SEQUORA data")
    parser.add_argument("--processed-dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-train-users", type=int, default=50000,
                         help="Subsample this many users for training (CPU-friendly). "
                              "Set to a very large number to use everyone.")
    parser.add_argument("--eval-users", type=int, default=2000,
                         help="Subsample this many users for evaluation each epoch.")
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
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