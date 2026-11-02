"""
SEQUORA — Catalogue helpers
=============================
One place that decides which category an article belongs to, so the API
(backend category filtering), the persona script and the Streamlit app
all agree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

CATEGORY_KEYWORDS = {
    "Dresses": ["dress"],
    "Tops": ["shirt", "top", "blouse", "t-shirt", "tee", "sweater", "jumper", "bodysuit", "vest", "hoodie", "cardigan"],
    "Trousers": ["trouser", "pant", "jean", "legging", "short"],
    "Footwear": ["shoe", "sneaker", "sandal", "boot", "heel", "slipper", "pump"],
    "Outerwear": ["jacket", "coat", "parka", "blazer"],
    "Accessories": ["bag", "belt", "scarf", "hat", "jewellery", "sunglasses", "wallet", "glove", "earring", "necklace", "sock"],
    "Activewear": ["sport", "active", "gym", "training", "track"],
}
SWIM_KEYWORDS = ["swim", "bikini", "swimsuit", "swimwear"]


def classify_category(product_type_name: str, product_group_name: str = "", section_name: str = "") -> Optional[str]:
    """Use H&M's own product group first; fall back to keywords on the type name.
    Returns None for swimwear (excluded from the catalogue)."""
    ptype = (product_type_name or "").lower()
    group = (product_group_name or "").strip().lower()
    section = (section_name or "").lower()

    if group == "swimwear" or any(kw in ptype for kw in SWIM_KEYWORDS):
        return None

    # H&M's data has a few lingerie items mislabelled as Slippers/Shoes.
    # The section name is reliable, so lingerie always wins.
    if "lingerie" in section:
        return "Underwear"

    if group:
        if group == "shoes":
            return "Footwear"
        if group in ("underwear", "nightwear"):
            return "Underwear"
        if group in ("socks & tights", "accessories", "bags", "cosmetic", "fun", "items", "stationery"):
            return "Accessories"
        if any(kw in ptype for kw in CATEGORY_KEYWORDS["Activewear"]):
            return "Activewear"
        if group == "garment full body":
            return "Dresses" if "dress" in ptype else "Other"
        if group == "garment lower body":
            return "Other" if "skirt" in ptype else "Trousers"
        if group == "garment upper body":
            if any(kw in ptype for kw in ["jacket", "coat", "parka", "blazer", "gilet"]):
                return "Outerwear"
            return "Tops"
        return "Other"

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in ptype for kw in keywords):
            return category
    return "Other"


def build_item_categories(processed_dir: Path) -> List[Optional[str]]:
    """List indexed by internal item index -> category name (None = excluded swimwear / padding)."""
    processed_dir = Path(processed_dir)
    id_map = json.loads((processed_dir / "article_id_map.json").read_text())
    articles = pd.read_parquet(processed_dir / "articles_clean.parquet")
    info = articles.set_index("article_id").to_dict(orient="index")

    size = max(int(v) for v in id_map.values()) + 1
    cats: List[Optional[str]] = ["Other"] * size
    cats[0] = None  # padding index
    for article_id, idx in id_map.items():
        row = info.get(article_id, {})
        cats[int(idx)] = classify_category(
            row.get("product_type_name", ""),
            row.get("product_group_name", ""),
            row.get("section_name", ""),
        )
    return cats