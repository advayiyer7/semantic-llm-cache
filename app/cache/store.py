"""Vector store backed by Redis Stack (RedisVL).

One Redis Stack container provides both the vector index (HNSW + cosine) and
TTL-based eviction. Entries are isolated per-namespace via a tag filter so a
KNN search only ever matches requests with the same model/system/params.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag
from redisvl.schema import IndexSchema


@dataclass(frozen=True)
class CacheHit:
    """A nearest-neighbour match: the stored prompt/response and its similarity."""

    prompt: str
    response: str
    similarity: float


class CacheStore:
    def __init__(
        self,
        redis_url: str,
        index_name: str,
        index_prefix: str,
        dims: int,
    ) -> None:
        # Schema: a tag field for namespace isolation + a cosine HNSW vector field.
        # Hash storage keeps each entry a single Redis hash with a native TTL.
        self._schema = IndexSchema.from_dict(
            {
                "index": {
                    "name": index_name,
                    "prefix": index_prefix,
                    "storage_type": "hash",
                },
                "fields": [
                    {"name": "namespace", "type": "tag"},
                    {"name": "prompt", "type": "text"},
                    {"name": "response", "type": "text"},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "attrs": {
                            "dims": dims,
                            "distance_metric": "cosine",
                            "algorithm": "hnsw",
                            "datatype": "float32",
                        },
                    },
                ],
            }
        )
        self._index = SearchIndex(self._schema, redis_url=redis_url)

    def ensure_index(self) -> None:
        """Create the index if missing (idempotent)."""
        self._index.create(overwrite=False)

    def ping(self) -> bool:
        """True if Redis is reachable (drives the readiness probe)."""
        try:
            return bool(self._index.client.ping())
        except Exception:  # noqa: BLE001
            return False

    def search(self, namespace: str, vector: list[float], k: int = 1) -> CacheHit | None:
        """Nearest neighbour within a namespace. Similarity = 1 - cosine distance."""
        query = VectorQuery(
            vector=vector,
            vector_field_name="embedding",
            return_fields=["namespace", "prompt", "response"],
            num_results=k,
            filter_expression=Tag("namespace") == namespace,
        )
        results = self._index.query(query)
        if not results:
            return None

        top = results[0]
        distance = float(top["vector_distance"])
        return CacheHit(
            prompt=top.get("prompt", ""),
            response=top.get("response", ""),
            similarity=1.0 - distance,
        )

    def store(
        self,
        namespace: str,
        vector: list[float],
        prompt: str,
        response: str,
        ttl: int,
    ) -> None:
        """Insert one cache entry (embedding + prompt + response) under a TTL."""
        record = {
            "namespace": namespace,
            "prompt": prompt,
            "response": response,
            "embedding": np.asarray(vector, dtype=np.float32).tobytes(),
        }
        self._index.load([record], ttl=ttl)

    def drop(self) -> None:
        """Delete the index and all its keys (used in tests)."""
        self._index.delete(drop=True)
