"""Semantic cache engine — ties embedding, namespacing, and vector search.

`query()` embeds the prompt exactly once and returns the embedding alongside the
hit/miss result, so the proxy can reuse that vector to `store()` a miss without
paying for a second embedding call (which would inflate the latency budget).
"""

from __future__ import annotations

from app.cache.embedder import Embedder
from app.cache.keys import cache_namespace
from app.cache.store import CacheHit, CacheStore


class QueryResult:
    """Outcome of a cache lookup. `hit` is None on a miss."""

    __slots__ = ("namespace", "vector", "hit")

    def __init__(self, namespace: str, vector: list[float], hit: CacheHit | None) -> None:
        self.namespace = namespace
        self.vector = vector
        self.hit = hit


class SemanticCache:
    def __init__(
        self,
        embedder: Embedder,
        store: CacheStore,
        threshold: float,
        default_ttl: int,
    ) -> None:
        self.embedder = embedder
        self._store = store
        self._threshold = threshold
        self._default_ttl = default_ttl

    def query(
        self,
        model: str,
        system_prompt: str | None,
        params: dict | None,
        user_prompt: str,
        threshold: float | None = None,
    ) -> QueryResult:
        namespace = cache_namespace(model, system_prompt, params)
        vector = self.embedder.embed(user_prompt)
        hit = self._store.search(namespace, vector, k=1)
        limit = self._threshold if threshold is None else threshold
        if hit is not None and hit.similarity >= limit:
            return QueryResult(namespace, vector, hit)
        return QueryResult(namespace, vector, None)

    def lookup(self, namespace: str, vector: list[float], threshold: float | None = None):
        """Re-check the cache with an already-computed vector (no re-embedding)."""
        hit = self._store.search(namespace, vector, k=1)
        limit = self._threshold if threshold is None else threshold
        if hit is not None and hit.similarity >= limit:
            return hit
        return None

    def store(
        self,
        namespace: str,
        vector: list[float],
        user_prompt: str,
        response: str,
        ttl: int | None = None,
    ) -> None:
        self._store.store(
            namespace, vector, user_prompt, response, ttl or self._default_ttl
        )
