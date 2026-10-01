"""
SEQUORA — Multimodal Cold-Start Encoder
==========================================
Gives every product a vector BEFORE anyone has interacted with it, using
only its photo and its text metadata. This is what lets a brand-new item
be recommended immediately — no purchase history required.

Two encoders, projected into one shared space:
  ImageEncoder  — pretrained ResNet-50, final classification layer removed
  TextEncoder   — sentence-transformers MiniLM on product text
  MultimodalEncoder — combines both with a small learned projection layer

Run this file directly to encode every article in articles_clean.parquet
and save the resulting vectors to disk.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

logger = logging.getLogger("sequora.multimodal")

IMAGE_EMBED_DIM = 2048   # ResNet-50's final feature size
TEXT_EMBED_DIM = 384     # all-MiniLM-L6-v2's output size


class ImageEncoder(nn.Module):
    """Pretrained ResNet-50 with the classification head removed, so the
    output is a 2048-dim feature vector instead of class probabilities."""

    def __init__(self) -> None:
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # drop fc layer
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def encode_image(self, image_path: Path) -> np.ndarray:
        """Encodes one image. Returns a zero vector if the file is missing
        or unreadable, so a bad/missing photo never crashes the whole run."""
        try:
            img = Image.open(image_path).convert("RGB")
            tensor = self.preprocess(img).unsqueeze(0)
            features = self.backbone(tensor)
            return features.flatten().numpy()
        except (FileNotFoundError, OSError) as exc:
            logger.warning("Could not read image %s (%s) — using zero vector", image_path, exc)
            return np.zeros(IMAGE_EMBED_DIM, dtype=np.float32)


class TextEncoder:
    """Wraps sentence-transformers for product text (name + category + description)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode_text(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )


class MultimodalEncoder(nn.Module):
    """
    Projects concatenated [image_vec, text_vec] into one shared item-content
    space. This projected vector is what later feeds the Two-Tower item
    tower and the FAISS index.
    """

    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        input_dim = IMAGE_EMBED_DIM + TEXT_EMBED_DIM
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )

    def forward(self, image_vec: torch.Tensor, text_vec: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([image_vec, text_vec], dim=-1)
        projected = self.projection(combined)
        return nn.functional.normalize(projected, p=2, dim=-1)  # unit length, for cosine similarity


def resolve_image_path(raw_dir: Path, image_path: str) -> Path:
    return raw_dir / image_path


def encode_catalog(
    articles: pd.DataFrame,
    raw_dir: Path,
    image_encoder: ImageEncoder,
    text_encoder: TextEncoder,
    multimodal: MultimodalEncoder,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    if limit is not None:
        articles = articles.head(limit).copy()

    logger.info("Encoding text for %d articles...", len(articles))
    text_vectors = text_encoder.encode_text(articles["text_blob"].tolist())

    logger.info("Encoding images for %d articles (this is the slow part)...", len(articles))
    image_vectors = np.zeros((len(articles), IMAGE_EMBED_DIM), dtype=np.float32)
    for i, image_path in enumerate(tqdm(articles["image_path"], desc="Images")):
        full_path = resolve_image_path(raw_dir, image_path)
        image_vectors[i] = image_encoder.encode_image(full_path)

    with torch.no_grad():
        image_tensor = torch.tensor(image_vectors, dtype=torch.float32)
        text_tensor = torch.tensor(text_vectors, dtype=torch.float32)
        fused = multimodal(image_tensor, text_tensor).numpy()

    result = pd.DataFrame({
        "article_id": articles["article_id"].values,
    })
    for dim in range(fused.shape[1]):
        result[f"dim_{dim}"] = fused[:, dim]
    return result


def run(args: argparse.Namespace) -> None:
    articles = pd.read_parquet(args.processed_dir / "articles_clean.parquet")
    logger.info("Loaded %d articles", len(articles))

    image_encoder = ImageEncoder()
    text_encoder = TextEncoder()
    multimodal = MultimodalEncoder(output_dim=args.output_dim)

    embeddings = encode_catalog(
        articles, args.raw_dir, image_encoder, text_encoder, multimodal,
        limit=args.limit,
    )

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.artifacts_dir / "item_content_embeddings.parquet"
    embeddings.to_parquet(out_path, index=False)
    logger.info("Saved %d item embeddings (dim=%d) to %s", len(embeddings), args.output_dim, out_path)

    torch.save(multimodal.state_dict(), args.artifacts_dir / "multimodal_encoder.pt")
    logger.info("Saved multimodal encoder weights")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode H&M catalog with image+text embeddings")
    parser.add_argument("--processed-dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--raw-dir", type=Path, default=Path("raw_data/hm"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--limit", type=int, default=2000,
                         help="Encode only this many articles first (images are slow on CPU). "
                              "Raise this later once you confirm it works.")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(42)
    run(parse_args())