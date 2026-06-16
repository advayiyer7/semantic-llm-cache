"""Cache precision/recall over a labeled eval set.

Embeds each (query, candidate) pair with the configured backend, computes cosine
similarity, and sweeps thresholds — reporting hit-rate vs precision/recall/F1 so
the chosen similarity threshold is defensible, not guessed.

Usage:
    uv run python -m eval.precision
    EMBEDDING_BACKEND=local uv run python -m eval.precision
"""

from __future__ import annotations

import json
from pathlib import Path

from app.cache.embedder import build_embedder
from app.cache.tuner import cosine, sweep
from app.config import get_settings

_DATASET = Path(__file__).with_name("dataset.jsonl")
_THRESHOLDS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]


def main() -> None:
    settings = get_settings()
    embedder = build_embedder(settings)
    if embedder is None:
        raise SystemExit("No embedder configured (set OPENAI_API_KEY or EMBEDDING_BACKEND=local).")

    pairs = [json.loads(line) for line in _DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    scored = [
        (cosine(embedder.embed(p["query"]), embedder.embed(p["candidate"])), bool(p["should_hit"]))
        for p in pairs
    ]

    rows = sweep(scored, _THRESHOLDS)
    print(f"backend={settings.embedding_backend}  pairs={len(pairs)}\n")
    print(f"{'thresh':>7} {'hit_rate':>9} {'precision':>10} {'recall':>7} {'f1':>6}")
    for r in rows:
        print(f"{r['threshold']:>7} {r['hit_rate']:>9} {r['precision']:>10} {r['recall']:>7} {r['f1']:>6}")

    configured = next((r for r in rows if abs(r["threshold"] - settings.similarity_threshold) < 1e-9), None)
    best = max(rows, key=lambda r: (r["f1"], r["threshold"]))
    print(f"\nrecommended threshold (max F1): {best['threshold']}  (precision={best['precision']}, recall={best['recall']})")
    if configured:
        print(
            f"configured threshold {settings.similarity_threshold}: "
            f"precision={configured['precision']}, recall={configured['recall']}"
        )


if __name__ == "__main__":
    main()
