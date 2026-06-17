"""Deterministic test doubles."""

from __future__ import annotations

import asyncio
import hashlib
from typing import AsyncIterator

import httpx
import numpy as np

from app.proxy.openai_format import completion_response
from app.proxy.schemas import ChatCompletionRequest


class FailingStreamProvider:
    """Raises a provider-class error as soon as streaming starts."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, req: ChatCompletionRequest) -> dict:
        self.calls += 1
        raise httpx.RequestError("boom")

    async def stream(self, req: ChatCompletionRequest) -> AsyncIterator[str]:
        self.calls += 1
        if False:  # make this an async generator
            yield ""
        raise httpx.RequestError("boom")


class FakeProvider:
    """Records call count and echoes the last message — no network."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, req: ChatCompletionRequest) -> dict:
        self.calls += 1
        last = req.messages[-1].content
        return completion_response(req.model, f"FAKE:{last}")

    async def stream(self, req: ChatCompletionRequest) -> AsyncIterator[str]:
        self.calls += 1
        last = req.messages[-1].content
        for token in ("FAKE", ":", str(last)):
            yield token


class SlowFakeProvider:
    """Like FakeProvider but `complete` awaits a delay — opens a race window so
    single-flight behaviour is observable under concurrency."""

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.calls = 0

    async def complete(self, req: ChatCompletionRequest) -> dict:
        self.calls += 1
        await asyncio.sleep(self.delay)
        return completion_response(req.model, f"FAKE:{req.messages[-1].content}")

    async def stream(self, req: ChatCompletionRequest) -> AsyncIterator[str]:
        self.calls += 1
        await asyncio.sleep(self.delay)
        yield f"FAKE:{req.messages[-1].content}"


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
