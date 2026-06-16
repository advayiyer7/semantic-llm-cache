"""Phase 1 critical-path overhead benchmark.

Measures the latency the cache adds to every request: embedding + vector search.
This MUST stay well below provider latency, or cache misses go net-negative.

Requires a real OPENAI_API_KEY (in .env) and a running Redis Stack:
    docker-compose up -d redis
    uv run python scripts/bench_phase1.py
"""

from __future__ import annotations

import statistics
import time

from app.cache.embedder import OpenAIEmbedder
from app.cache.store import CacheStore
from app.config import get_settings

N = 50
NAMESPACE = "bench"


def _p95(samples: list[float]) -> float:
    return statistics.quantiles(samples, n=100)[94]


def main() -> None:
    settings = get_settings()
    store = CacheStore(
        settings.redis_url,
        settings.index_name,
        settings.index_prefix,
        settings.embedding_dim,
    )
    store.ensure_index()
    embedder = OpenAIEmbedder(settings)

    prompts = [f"sample query {i} about topic {i % 7}" for i in range(N)]

    # Seed the index so search has something to match against.
    for prompt in prompts[:10]:
        store.store(NAMESPACE, embedder.embed(prompt), prompt, "seed", ttl=300)

    embed_ms: list[float] = []
    search_ms: list[float] = []
    for prompt in prompts:
        t0 = time.perf_counter()
        vector = embedder.embed(prompt)
        embed_ms.append((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        store.search(NAMESPACE, vector)
        search_ms.append((time.perf_counter() - t1) * 1000)

    total = [e + s for e, s in zip(embed_ms, search_ms)]
    print(f"embed   P50={statistics.median(embed_ms):6.1f}ms  P95={_p95(embed_ms):6.1f}ms")
    print(f"search  P50={statistics.median(search_ms):6.1f}ms  P95={_p95(search_ms):6.1f}ms")
    print(f"overall P50={statistics.median(total):6.1f}ms  P95={_p95(total):6.1f}ms")


if __name__ == "__main__":
    main()
