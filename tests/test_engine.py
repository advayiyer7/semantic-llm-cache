"""Cache engine behaviour — integration tests against Redis Stack.

Covers the Phase 1 validation criteria:
  - exact repeat hits
  - unrelated prompt misses
  - param change misses (namespace isolation)
  - semantically similar (paraphrase) prompt hits
"""

from __future__ import annotations

import numpy as np

from app.cache.engine import SemanticCache

MODEL = "gpt-4o"
PARAMS = {"temperature": 0.0}


def _cache(embedder, store) -> SemanticCache:
    return SemanticCache(embedder, store, threshold=0.95, default_ttl=60)


def test_exact_repeat_hits(store, embedder):
    cache = _cache(embedder, store)
    prompt = "What is the capital of France?"

    miss = cache.query(MODEL, None, PARAMS, prompt)
    assert miss.hit is None
    cache.store(miss.namespace, miss.vector, prompt, "Paris")

    again = cache.query(MODEL, None, PARAMS, prompt)
    assert again.hit is not None
    assert again.hit.response == "Paris"
    assert again.hit.similarity >= 0.99


def test_unrelated_prompt_misses(store, embedder):
    cache = _cache(embedder, store)
    seed = "What is the capital of France?"
    res = cache.query(MODEL, None, PARAMS, seed)
    cache.store(res.namespace, res.vector, seed, "Paris")

    other = cache.query(MODEL, None, PARAMS, "Write a poem about the ocean tides")
    assert other.hit is None


def test_param_change_misses(store, embedder):
    cache = _cache(embedder, store)
    prompt = "What is the capital of France?"
    res = cache.query(MODEL, None, {"temperature": 0.0}, prompt)
    cache.store(res.namespace, res.vector, prompt, "Paris")

    # Same prompt, different temperature -> different namespace -> miss.
    hot = cache.query(MODEL, None, {"temperature": 0.9}, prompt)
    assert hot.hit is None


def test_paraphrase_hits(store, embedder):
    cache = _cache(embedder, store)
    original = "What is the capital of France?"
    paraphrase = "Tell me France's capital city"

    # Craft the paraphrase vector to sit very close to the original.
    base = np.asarray(embedder.embed(original))
    nudged = base + 0.01 * np.random.default_rng(1).standard_normal(base.shape[0])
    nudged /= np.linalg.norm(nudged)
    embedder.set_vector(paraphrase, nudged.tolist())

    res = cache.query(MODEL, None, PARAMS, original)
    cache.store(res.namespace, res.vector, original, "Paris")

    similar = cache.query(MODEL, None, PARAMS, paraphrase)
    assert similar.hit is not None
    assert similar.hit.response == "Paris"
