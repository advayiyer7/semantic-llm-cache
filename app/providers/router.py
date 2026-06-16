"""Route a request to a provider by its `model` field."""

from __future__ import annotations

_OPENAI_PREFIXES = ("gpt", "o1", "o3", "o4", "chatgpt", "text-")


def provider_key_for_model(model: str) -> str:
    """Map a model name to a provider key: openai | anthropic | ollama."""
    name = model.lower()
    if name.startswith(_OPENAI_PREFIXES):
        return "openai"
    if name.startswith("claude"):
        return "anthropic"
    # Everything else (llama3.2, mistral, qwen, ...) is served locally by Ollama.
    return "ollama"
