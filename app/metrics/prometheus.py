"""Prometheus metrics + the near-miss analyzer.

Exposes request/hit-rate counters, cached-vs-uncached latency histograms, a
similarity-score distribution, and an estimated-cost-saved counter. The
near-miss analyzer keeps a bounded in-memory log of lookups that landed just
below the threshold — the data you use to decide whether to loosen it.
"""

from __future__ import annotations

import threading
from collections import deque

from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "semantic_cache_requests_total",
    "Proxy chat-completion requests by cache result.",
    ["result"],
)
LATENCY = Histogram(
    "semantic_cache_request_latency_seconds",
    "Non-streaming request latency by cache result.",
    ["result"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
SIMILARITY = Histogram(
    "semantic_cache_similarity",
    "Top-1 similarity observed on cache lookups.",
    buckets=(0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0),
)
COST_SAVED = Counter(
    "semantic_cache_cost_saved_usd_total",
    "Estimated USD saved by serving cache hits instead of calling the provider.",
)
NEAR_MISS = Counter(
    "semantic_cache_near_miss_total",
    "Lookups that missed but landed just below the threshold.",
)

_near_misses: deque[dict] = deque(maxlen=100)
_lock = threading.Lock()


def record_request(result: str, latency_seconds: float | None = None) -> None:
    REQUESTS.labels(result=result).inc()
    if latency_seconds is not None:
        LATENCY.labels(result=result).observe(latency_seconds)


def observe_similarity(similarity: float) -> None:
    SIMILARITY.observe(similarity)


def record_cost_saved(usd: float) -> None:
    if usd > 0:
        COST_SAVED.inc(usd)


def record_near_miss(namespace: str, similarity: float, threshold: float) -> None:
    NEAR_MISS.inc()
    with _lock:
        _near_misses.append(
            {
                "namespace": namespace,
                "similarity": round(similarity, 4),
                "threshold": threshold,
            }
        )


def recent_near_misses() -> list[dict]:
    with _lock:
        return list(_near_misses)


def estimate_saved_usd(prompt_text: str, response_text: str, price_per_1k_tokens: float) -> float:
    """Rough estimate: ~4 chars/token, priced at a blended rate."""
    approx_tokens = (len(prompt_text) + len(response_text)) / 4.0
    return approx_tokens / 1000.0 * price_per_1k_tokens
