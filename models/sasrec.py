"""
SEQUORA — SASRec (Self-Attentive Sequential Recommendation)
=============================================================
Reads a user's ordered history of items and predicts which item comes next.

Input conventions (match data/dataset.py exactly):
  * item index 0 is PADDING, real items are 1..num_items
  * sequences are padded on the LEFT, so the most recent item is always
    the last position

How it works, in plain terms:
  1. Every item gets a learned vector (embedding), plus a vector for its
     position in the sequence.
  2. A stack of self-attention blocks lets each position look at the
     items BEFORE it (never after it) to decide what matters.
  3. The output at each position is compared with candidate item vectors
     using a dot product. High score = likely next item.
"""

from __future__ import annotations

import argparse
import logging
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("sequora.sasrec")


class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SASRecBlock(nn.Module):
    """One transformer block: causal self-attention, then a feed-forward layer."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = PointWiseFeedForward(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        timeline_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.attn_norm(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.ffn_norm(x))
        # zero out padded positions so padding never carries information
        return x * timeline_mask


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        d_model: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_items = num_items
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads

        # index 0 is reserved for padding
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [SASRecBlock(d_model, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.final_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def _build_attn_mask(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Boolean mask, True = "not allowed to attend".
        Blocks (a) looking at future positions and (b) looking at padding.
        The diagonal is always allowed so no row is fully blocked, which
        would otherwise produce NaN values inside the softmax.
        """
        batch_size, seq_len = input_seq.shape
        device = input_seq.device

        causal = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )
        pad_keys = (input_seq == 0).unsqueeze(1)
        blocked = causal.unsqueeze(0) | pad_keys
        eye = torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0)
        blocked = blocked & ~eye
        return blocked.repeat_interleave(self.num_heads, dim=0)

    def encode(self, input_seq: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len) item ids -> (batch, seq_len, d_model) hidden states."""
        _, seq_len = input_seq.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")

        positions = torch.arange(seq_len, device=input_seq.device).unsqueeze(0)
        x = self.item_emb(input_seq) * math.sqrt(self.d_model) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        timeline_mask = (input_seq != 0).unsqueeze(-1).to(x.dtype)
        x = x * timeline_mask
        attn_mask = self._build_attn_mask(input_seq)

        for block in self.blocks:
            x = block(x, attn_mask, timeline_mask)
        return self.final_norm(x)

    def forward(
        self,
        input_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        neg_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode(input_seq)
        pos_logits = (hidden * self.item_emb(pos_seq)).sum(dim=-1)
        neg_logits = (hidden * self.item_emb(neg_seq)).sum(dim=-1)
        return pos_logits, neg_logits

    def compute_loss(
        self,
        input_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        neg_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy: push true next items up, sampled negatives down."""
        pos_logits, neg_logits = self.forward(input_seq, pos_seq, neg_seq)
        valid = pos_seq != 0
        if not valid.any():
            return pos_logits.sum() * 0.0

        pos_valid = pos_logits[valid]
        neg_valid = neg_logits[valid]
        pos_loss = F.binary_cross_entropy_with_logits(pos_valid, torch.ones_like(pos_valid))
        neg_loss = F.binary_cross_entropy_with_logits(neg_valid, torch.zeros_like(neg_valid))
        return pos_loss + neg_loss

    def user_embedding(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Summary vector of the user's recent activity (last position)."""
        return self.encode(input_seq)[:, -1, :]

    @torch.no_grad()
    def score_all_items(self, input_seq: torch.Tensor) -> torch.Tensor:
        """(batch, num_items + 1) scores; padding index 0 is never recommendable."""
        user_vec = self.user_embedding(input_seq)
        scores = user_vec @ self.item_emb.weight.T
        scores[:, 0] = float("-inf")
        return scores

    @torch.no_grad()
    def recommend(
        self, input_seq: torch.Tensor, k: int = 10, exclude_seen: bool = True
    ) -> torch.Tensor:
        """Top-k item ids per user, optionally skipping items already in their history."""
        scores = self.score_all_items(input_seq)
        if exclude_seen:
            scores.scatter_(1, input_seq, float("-inf"))
        return scores.topk(k, dim=1).indices


def _smoke_test(num_items: int, batch_size: int, max_len: int) -> None:
    torch.manual_seed(42)
    random.seed(42)

    model = SASRec(num_items=num_items, max_len=max_len)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model built. Parameters: {n_params:,}")

    # fake batch shaped exactly like SASRecDataset output, with random left-padding
    input_seq = torch.randint(1, num_items + 1, (batch_size, max_len))
    pos_seq = torch.randint(1, num_items + 1, (batch_size, max_len))
    neg_seq = torch.randint(1, num_items + 1, (batch_size, max_len))
    for row in range(batch_size):
        pad_len = random.randint(0, max_len - 3)
        input_seq[row, :pad_len] = 0
        pos_seq[row, :pad_len] = 0
        neg_seq[row, :pad_len] = 0

    # 1. one training step: loss is finite and gradients flow
    model.train()
    loss = model.compute_loss(input_seq, pos_seq, neg_seq)
    loss.backward()
    assert torch.isfinite(loss), "Loss is not finite"
    assert model.item_emb.weight.grad is not None, "No gradient reached item embeddings"
    print(f"[1/4] Training step OK. Loss = {loss.item():.4f}")

    # 2. encoder output has the right shape and no NaN values
    model.eval()
    with torch.no_grad():
        hidden = model.encode(input_seq)
    assert hidden.shape == (batch_size, max_len, model.d_model)
    assert torch.isfinite(hidden).all(), "Encoder produced NaN or inf"
    print(f"[2/4] Encoder output OK. Shape = {tuple(hidden.shape)}")

    # 3. causality: changing the LAST item must not change any earlier position
    with torch.no_grad():
        changed = input_seq.clone()
        changed[:, -1] = (input_seq[:, -1] % num_items) + 1
        hidden_changed = model.encode(changed)
    assert torch.allclose(hidden[:, :-1], hidden_changed[:, :-1], atol=1e-5), (
        "Model is leaking future information"
    )
    print("[3/4] Causality OK. Earlier positions never see later items.")

    # 4. recommendations: right shape, no padding id, no already-seen items
    with torch.no_grad():
        top_items = model.recommend(input_seq, k=10)
    assert top_items.shape == (batch_size, 10)
    assert (top_items != 0).all(), "Padding index was recommended"
    for row in range(batch_size):
        seen = set(input_seq[row].tolist())
        assert not (set(top_items[row].tolist()) & seen), "Recommended an already-seen item"
    print("[4/4] Recommendations OK. Top-10 per user, no padding, no repeats.")

    print("SASREC SMOKE TEST PASSED")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="SASRec smoke test")
    parser.add_argument("--num-items", type=int, default=105542)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=50)
    args = parser.parse_args()

    _smoke_test(args.num_items, args.batch_size, args.max_len)