"""30-second demo — show the semantic cache in action.

Fires four prompts at the running proxy and narrates the cache behaviour:
  1. a fresh question         -> MISS  (the provider is actually called)
  2. the exact same question  -> HIT   (served from cache, ~instant)
  3. a paraphrase             -> HIT   (semantic match — different words, same meaning)
  4. an unrelated question    -> MISS

A unique per-run system prompt isolates this demo into its own cache namespace,
so step 1 is always a clean miss regardless of what's already cached.

Run the proxy first (fully local, no API key needed):
    docker compose up -d redis
    uv sync --group local
    EMBEDDING_BACKEND=local THRESHOLD_BALANCED=0.85 uv run uvicorn app.main:app
Then:
    uv run python scripts/demo.py
"""

from __future__ import annotations

import argparse
import time

import httpx

_STEPS = [
    ("fresh question", "What is the capital of France?"),
    ("exact repeat", "What is the capital of France?"),
    ("paraphrase - different words, same meaning", "Can you tell me France's capital city?"),
    ("unrelated question", "Write a one-line haiku about the ocean."),
]


def _ask(client: httpx.Client, model: str, system: str, prompt: str):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 32,
    }
    start = time.perf_counter()
    resp = client.post("/v1/chat/completions", json=payload)
    elapsed_ms = (time.perf_counter() - start) * 1000
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"].strip().replace("\n", " ")
    return resp.headers.get("X-Cache", "?"), elapsed_ms, answer


def _scrape_cost_saved(client: httpx.Client) -> float:
    try:
        for line in client.get("/metrics").text.splitlines():
            if line.startswith("semantic_cache_cost_saved_usd_total "):
                return float(line.split()[1])
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="llama3.2:3b")
    args = parser.parse_args()

    system = f"You are a concise assistant. (demo-session-{int(time.time())})"
    print(f"\n  Semantic LLM Cache - live demo  (model: {args.model})")
    print("  " + "-" * 64)

    with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        for label, prompt in _STEPS:
            cache, elapsed_ms, answer = _ask(client, args.model, system, prompt)
            tag = {"HIT": "HIT ", "MISS": "MISS"}.get(cache, cache)
            print(f"\n  [{tag}]  {elapsed_ms:8.1f} ms   {label}")
            print(f"           prompt: {prompt}")
            print(f"           answer: {answer[:78]}")

        print("\n  " + "-" * 64)
        print(f"  estimated cost saved so far: ${_scrape_cost_saved(client):.6f}")
        print("  (exact + paraphrase both served from cache without calling the model)\n")


if __name__ == "__main__":
    main()
