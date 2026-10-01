"""
SEQUORA — FAISS Retrieval Index
==================================
Wraps FAISS so item vectors (from the Two-Tower item tower) can be
searched instantly, and so a brand-new item can be inserted into the
live index without rebuilding anything — this is what makes real-time
cold-start actually work, not just in theory.

Uses IndexIDMap wrapped around a flat inner-product index: vectors are
unit-length (Two-Tower normalizes them), so inner product = cosine
similarity, and IndexIDMap lets us add_with_ids() new items on the fly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

logger = logging.getLogger("sequora.faiss_index")


class FaissItemIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        base_index = faiss.IndexFlatIP(dim)   # inner product = cosine similarity, since vectors are unit-length
        self.index = faiss.IndexIDMap(base_index)
        self._id_set = set()

    def add(self, item_ids: np.ndarray, vectors: np.ndarray) -> None:
        """
        item_ids: (n,) int64 array of item ids
        vectors:  (n, dim) float32 array, should already be unit-length
        Adding an id that already exists creates a duplicate entry in
        FAISS (it doesn't overwrite) — remove_ids first if you need to
        replace an existing item's vector.
        """
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if item_ids.dtype != np.int64:
            item_ids = item_ids.astype(np.int64)

        self.index.add_with_ids(vectors, item_ids)
        self._id_set.update(item_ids.tolist())
        logger.info("Added %d vectors. Index now holds %d total.", len(item_ids), self.index.ntotal)

    def add_one(self, item_id: int, vector: np.ndarray) -> None:
        """Convenience wrapper for the live 'new item uploaded' path —
        this is the exact call app/services.py makes from POST /index/add."""
        self.add(np.array([item_id]), vector.reshape(1, -1))

    def remove(self, item_ids: List[int]) -> int:
        """Removes items by id (e.g. delisted products). Returns how many
        were actually removed."""
        id_selector = faiss.IDSelectorBatch(np.array(item_ids, dtype=np.int64))
        n_removed = self.index.remove_ids(id_selector)
        self._id_set.difference_update(item_ids)
        logger.info("Removed %d vectors. Index now holds %d total.", n_removed, self.index.ntotal)
        return n_removed

    def search(self, query_vectors: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        query_vectors: (n_queries, dim) float32
        Returns (scores, item_ids), both shape (n_queries, k).
        item_ids of -1 mean "no result" (happens if the index has fewer
        than k items total).
        """
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        scores, item_ids = self.index.search(query_vectors, k)
        return scores, item_ids

    def contains(self, item_id: int) -> bool:
        return item_id in self._id_set

    @property
    def size(self) -> int:
        return self.index.ntotal

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))
        logger.info("Saved FAISS index (%d vectors) to %s", self.size, path)

    @classmethod
    def load(cls, path: Path, dim: int) -> "FaissItemIndex":
        wrapper = cls(dim)
        wrapper.index = faiss.read_index(str(path))
        id_selector_all = faiss.vector_to_array(wrapper.index.id_map)
        wrapper._id_set = set(id_selector_all.tolist())
        logger.info("Loaded FAISS index (%d vectors) from %s", wrapper.size, path)
        return wrapper


def _smoke_test(dim: int = 128, n_items: int = 500, n_queries: int = 5) -> None:
    rng = np.random.default_rng(42)

    def random_unit_vectors(n: int) -> np.ndarray:
        v = rng.normal(size=(n, dim)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    index = FaissItemIndex(dim=dim)

    # 1. build from scratch
    item_ids = np.arange(1, n_items + 1)
    vectors = random_unit_vectors(n_items)
    index.add(item_ids, vectors)
    assert index.size == n_items, f"Expected {n_items} vectors, got {index.size}"
    print(f"[1/5] Build OK. Index holds {index.size} vectors.")

    # 2. search returns a vector's own id as its top match (self-similarity = 1.0)
    query = vectors[0:1]
    scores, result_ids = index.search(query, k=5)
    assert result_ids[0, 0] == item_ids[0], "Top result should be the item itself"
    assert abs(scores[0, 0] - 1.0) < 1e-4, f"Self-similarity should be ~1.0, got {scores[0, 0]}"
    print(f"[2/5] Search OK. Self-match score = {scores[0, 0]:.4f}")

    # 3. incremental add — a brand-new item is searchable WITHOUT rebuilding
    new_item_id = 999_999
    new_vector = vectors[0].copy()  # make it nearly identical to item 1, on purpose
    index.add_one(new_item_id, new_vector)
    assert index.contains(new_item_id), "New item not found in index after add_one"
    scores2, result_ids2 = index.search(new_vector.reshape(1, -1), k=3)
    assert new_item_id in result_ids2[0], "Newly added item did not retrieve itself"
    print(f"[3/5] Incremental add OK. New item {new_item_id} is immediately searchable.")

    # 4. remove works
    removed = index.remove([new_item_id])
    assert removed == 1, f"Expected to remove 1 item, removed {removed}"
    assert not index.contains(new_item_id), "Item still marked as present after removal"
    print("[4/5] Remove OK. Item no longer in index.")

    # 5. save and reload preserves the index
    tmp_path = Path("artifacts/_faiss_smoke_test.index")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(tmp_path)
    reloaded = FaissItemIndex.load(tmp_path, dim=dim)
    assert reloaded.size == index.size, "Reloaded index has a different size"
    scores3, result_ids3 = reloaded.search(query, k=1)
    assert result_ids3[0, 0] == item_ids[0], "Reloaded index gives a different top result"
    tmp_path.unlink()
    print("[5/5] Save/load OK. Reloaded index matches the original.")

    print("FAISS INDEX SMOKE TEST PASSED")


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _smoke_test()