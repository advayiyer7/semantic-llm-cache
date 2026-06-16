"""Load test the proxy and report the portfolio headline numbers.

Fires a documented workload at the proxy, records client-side latency + the
X-Cache result per request, then computes hit rate, cached-vs-uncached latency
percentiles, and (from /metrics) estimated cost saved.

Usage:
    uv run python -m loadtest.run --n 2000 --concurrency 10 --model llama3.2:3b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx

from loadtest.workload import build_workload, mix_counts


def _pct(values: list[float], q: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100)[q - 1]


async def _fire(client, prompt, model, max_tokens):
    start = time.perf_counter()
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        },
    )
    return resp.headers.get("X-Cache", "?"), (time.perf_counter() - start), resp.status_code


async def run(args) -> dict:
    workload = build_workload(args.n)
    sem = asyncio.Semaphore(args.concurrency)
    print(f"workload mix: {mix_counts(workload)}  (n={args.n})")

    async with httpx.AsyncClient(base_url=args.base_url, timeout=180.0) as client:
        async def worker(item):
            async with sem:
                cache, latency, status = await _fire(
                    client, item["prompt"], args.model, args.max_tokens
                )
                return item["kind"], cache, latency, status

        wall_start = time.perf_counter()
        results = await asyncio.gather(*(worker(it) for it in workload))
        wall = time.perf_counter() - wall_start

        cost_saved = await _scrape_cost_saved(client)

    hit = [d for k, c, d, s in results if c == "HIT"]
    miss = [d for k, c, d, s in results if c == "MISS"]
    errors = [s for k, c, d, s in results if s >= 400]
    total = len(results)

    # Hits per workload kind. A "unique" that hits is, by definition, a false
    # positive — this is the empirical precision signal alongside the eval set.
    by_kind: dict[str, dict[str, int]] = {}
    for kind, cache, _d, _s in results:
        bucket = by_kind.setdefault(kind, {"hit": 0, "total": 0})
        bucket["total"] += 1
        if cache == "HIT":
            bucket["hit"] += 1
    uniques = by_kind.get("unique", {"hit": 0, "total": 0})
    false_hit_rate = round(uniques["hit"] / uniques["total"], 3) if uniques["total"] else 0.0

    summary = {
        "requests": total,
        "wall_seconds": round(wall, 1),
        "throughput_rps": round(total / wall, 1) if wall else 0,
        "hit_rate": round(len(hit) / total, 3) if total else 0,
        "errors": len(errors),
        "latency_ms": {
            "cached_p50": round(_pct(hit, 50) * 1000, 1),
            "cached_p95": round(_pct(hit, 95) * 1000, 1),
            "uncached_p50": round(_pct(miss, 50) * 1000, 1),
            "uncached_p95": round(_pct(miss, 95) * 1000, 1),
        },
        "estimated_cost_saved_usd": round(cost_saved, 6),
        "hits_by_kind": {k: f"{v['hit']}/{v['total']}" for k, v in by_kind.items()},
        "unique_false_hit_rate": false_hit_rate,
    }
    p95_cached = summary["latency_ms"]["cached_p95"]
    p95_uncached = summary["latency_ms"]["uncached_p95"]
    summary["p95_latency_reduction_pct"] = (
        round((1 - p95_cached / p95_uncached) * 100, 1) if p95_uncached else 0
    )
    return summary


async def _scrape_cost_saved(client) -> float:
    try:
        text = (await client.get("/metrics")).text
    except Exception:
        return 0.0
    for line in text.splitlines():
        if line.startswith("semantic_cache_cost_saved_usd_total "):
            return float(line.split()[1])
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--out", default="loadtest/results.json")
    args = parser.parse_args()

    summary = asyncio.run(run(args))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
