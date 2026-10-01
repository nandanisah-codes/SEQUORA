"""
SEQUORA — Two-Tower Retrieval Model
======================================
Two small encoders mapped into the SAME vector space:
  UserTower — a user's recent item sequence -> one vector
  ItemTower — an item's id + its multimodal content vector -> one vector

Trained with in-batch negative sampling (InfoNCE): for each (user, item)
pair in a batch, every OTHER item in that batch acts as a negative. This
is what makes Two-Tower training fast and is the standard approach used
by large-scale production recommenders.

At serving time, every item's tower output is precomputed once and
stored in FAISS (see retrieval/faiss_index.py) so retrieval is a fast
nearest-neighbor lookup instead of scoring the whole catalog per request.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("sequora.two_tower")


class UserTower(nn.Module):
    """Turns a user's recent item sequence into one embedding, via mean
    pooling over item embeddings followed by a small MLP projection."""

    def __init__(self, num_items: int, item_emb_dim: int = 64, output_dim: int = 128) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, item_emb_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(item_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """input_seq: (batch, seq_len) item ids, 0 = padding."""
        embedded = self.item_emb(input_seq)                       # (batch, seq_len, dim)
        mask = (input_seq != 0).unsqueeze(-1).float()              # (batch, seq_len, 1)
        summed = (embedded * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts                                   # mean over real (non-pad) items
        projected = self.mlp(pooled)
        return F.normalize(projected, p=2, dim=-1)


class ItemTower(nn.Module):
    """Turns an item id + its precomputed multimodal content vector into
    one embedding, in the same space as UserTower's output."""

    def __init__(
        self,
        num_items: int,
        content_dim: int = 128,
        item_emb_dim: int = 64,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, item_emb_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(item_emb_dim + content_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )

    def forward(self, item_ids: torch.Tensor, content_vecs: torch.Tensor) -> torch.Tensor:
        """
        item_ids: (batch,) item ids
        content_vecs: (batch, content_dim) from the multimodal encoder.
                      For a brand-new item with no learned id embedding yet,
                      pass item_id=0 (padding/unknown) — the content vector
                      still carries real signal, which is exactly what
                      makes cold-start work.
        """
        id_part = self.item_emb(item_ids)
        combined = torch.cat([id_part, content_vecs], dim=-1)
        projected = self.mlp(combined)
        return F.normalize(projected, p=2, dim=-1)


class TwoTower(nn.Module):
    def __init__(
        self,
        num_items: int,
        content_dim: int = 128,
        item_emb_dim: int = 64,
        output_dim: int = 128,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.user_tower = UserTower(num_items, item_emb_dim, output_dim)
        self.item_tower = ItemTower(num_items, content_dim, item_emb_dim, output_dim)
        self.temperature = temperature

    def forward(
        self,
        input_seq: torch.Tensor,
        item_ids: torch.Tensor,
        content_vecs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_vecs = self.user_tower(input_seq)
        item_vecs = self.item_tower(item_ids, content_vecs)
        return user_vecs, item_vecs

    def compute_loss(
        self,
        input_seq: torch.Tensor,
        item_ids: torch.Tensor,
        content_vecs: torch.Tensor,
    ) -> torch.Tensor:
        """
        In-batch InfoNCE: the similarity matrix's diagonal is the true
        positive pairs; every off-diagonal entry in the same row is an
        in-batch negative. Standard cross-entropy over that matrix.
        """
        user_vecs, item_vecs = self.forward(input_seq, item_ids, content_vecs)
        logits = (user_vecs @ item_vecs.T) / self.temperature   # (batch, batch)
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def encode_users(self, input_seq: torch.Tensor) -> torch.Tensor:
        return self.user_tower(input_seq)

    @torch.no_grad()
    def encode_items(self, item_ids: torch.Tensor, content_vecs: torch.Tensor) -> torch.Tensor:
        return self.item_tower(item_ids, content_vecs)


def _smoke_test(num_items: int, batch_size: int, max_len: int, content_dim: int) -> None:
    torch.manual_seed(42)

    model = TwoTower(num_items=num_items, content_dim=content_dim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model built. Parameters: {n_params:,}")

    input_seq = torch.randint(1, num_items + 1, (batch_size, max_len))
    item_ids = torch.randint(1, num_items + 1, (batch_size,))
    content_vecs = torch.randn(batch_size, content_dim)

    # 1. training step: loss finite, gradients flow into both towers
    model.train()
    loss = model.compute_loss(input_seq, item_ids, content_vecs)
    loss.backward()
    assert torch.isfinite(loss), "Loss is not finite"
    assert model.user_tower.item_emb.weight.grad is not None, "No gradient in user tower"
    assert model.item_tower.item_emb.weight.grad is not None, "No gradient in item tower"
    print(f"[1/4] Training step OK. Loss = {loss.item():.4f}")

    # 2. output vectors are unit-length (required for cosine similarity / FAISS IP index)
    model.eval()
    with torch.no_grad():
        user_vecs, item_vecs = model(input_seq, item_ids, content_vecs)
    user_norms = user_vecs.norm(dim=-1)
    item_norms = item_vecs.norm(dim=-1)
    assert torch.allclose(user_norms, torch.ones_like(user_norms), atol=1e-4), "User vectors not unit length"
    assert torch.allclose(item_norms, torch.ones_like(item_norms), atol=1e-4), "Item vectors not unit length"
    print(f"[2/4] Vector shapes OK. user={tuple(user_vecs.shape)}, item={tuple(item_vecs.shape)}, all unit-length")

    # 3. cold-start path: item_id=0 (unknown) still produces a valid, non-zero vector
    with torch.no_grad():
        cold_ids = torch.zeros(batch_size, dtype=torch.long)
        cold_content = torch.randn(batch_size, content_dim)
        cold_vecs = model.encode_items(cold_ids, cold_content)
    assert torch.isfinite(cold_vecs).all(), "Cold-start item produced NaN/inf"
    assert (cold_vecs.norm(dim=-1) > 0).all(), "Cold-start item produced a zero vector"
    print("[3/4] Cold-start item encoding OK. Unknown item id + content vector still works.")

    # 4. encode_users / encode_items match the forward pass exactly
    with torch.no_grad():
        u2 = model.encode_users(input_seq)
        i2 = model.encode_items(item_ids, content_vecs)
    assert torch.allclose(user_vecs, u2, atol=1e-6), "encode_users mismatch vs forward()"
    assert torch.allclose(item_vecs, i2, atol=1e-6), "encode_items mismatch vs forward()"
    print("[4/4] Standalone encode_users/encode_items match forward() exactly.")

    print("TWO-TOWER SMOKE TEST PASSED")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Two-Tower smoke test")
    parser.add_argument("--num-items", type=int, default=105542)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--content-dim", type=int, default=128)
    args = parser.parse_args()

    _smoke_test(args.num_items, args.batch_size, args.max_len, args.content_dim)