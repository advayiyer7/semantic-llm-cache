"""Critical-path overhead benchmark.

Measures the latency the cache adds to every request — embedding + vector search.
This must stay well below provider latency, or cache misses go net-negative.

Run with a Redis Stack up and any configured embedder (OpenAI or local):
    docker compose up -d redis
    uv run python scripts/bench_overhead.py
    EMBEDDING_BACKEND=local uv run python scripts/bench_overhead.py
"""

from __future__ import annotations

import statistics
import time

from app.cache.embedder import build_embedder
from app.cache.store import CacheStore
from app.config import get_settings

N = 50
NAMESPACE = "bench"


def _p95(samples: list[float]) -> float:
    return statistics.quantiles(samples, n=100)[94]


def main() -> None:
    settings = get_settings()
    embedder = build_embedder(settings)
    if embedder is None:
        raise SystemExit("No embedder configured (set OPENAI_API_KEY or EMBEDDING_BACKEND=local).")

    # Match the app's dimension-scoped index naming so the benchmark hits the
    # same index the proxy would use.
    store = CacheStore(
        settings.redis_url,
        f"{settings.index_name}_{embedder.dim}",
        f"{settings.index_prefix}_{embedder.dim}",
        embedder.dim,
    )
    store.ensure_index()

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
