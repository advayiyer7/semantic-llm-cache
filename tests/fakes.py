"""Deterministic test doubles."""

from __future__ import annotations

import hashlib

import numpy as np


class FakeEmbedder:
    """Deterministic embedder.

    Identical text always maps to the same unit vector (cosine similarity 1.0),
    and unrelated text maps to a near-orthogonal vector. `set_vector` lets a test
    craft a paraphrase that lands close to an existing entry.
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self._vectors: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        if text not in self._vectors:
            seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self._dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            self._vectors[text] = vec.tolist()
        return self._vectors[text]

    def set_vector(self, text: str, vector: list[float]) -> None:
        self._vectors[text] = list(vector)
