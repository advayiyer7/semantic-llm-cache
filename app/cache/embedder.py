"""Prompt embedding with a pluggable backend.

`build_embedder(settings)` selects the backend from `EMBEDDING_BACKEND`:
  - "openai": hosted `text-embedding-3-small` (needs OPENAI_API_KEY)
  - "local":  on-device `fastembed` (no key, no GPU, $0) — lets the whole proxy
              run fully local alongside Ollama.

The cache index dimensionality is taken from the embedder's `.dim`, so the two
never drift.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.config import Settings


@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    """Embeds prompts with OpenAI `text-embedding-3-small` (default)."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set — cannot create OpenAIEmbedder")
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model
        self._dim = settings.embedding_dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self._model, input=text)
        return response.data[0].embedding


class LocalEmbedder:
    """On-device embeddings via fastembed (ONNX, CPU). No API key required."""

    _DIMS = {"BAAI/bge-small-en-v1.5": 384}

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self._dim = self._DIMS.get(model_name, 384)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        # fastembed yields one numpy vector per input string.
        return next(self._model.embed([text])).tolist()


def build_embedder(settings: Settings) -> Embedder | None:
    """Return the configured embedder, or None when it can't be built."""
    backend = settings.embedding_backend.lower()
    if backend == "local":
        return LocalEmbedder(settings.local_embedding_model)
    if backend == "openai":
        if not settings.openai_api_key:
            return None
        return OpenAIEmbedder(settings)
    raise ValueError(f"Unknown EMBEDDING_BACKEND '{settings.embedding_backend}'.")
