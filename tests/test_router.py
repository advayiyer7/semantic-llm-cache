"""Provider routing — pure unit tests."""

from __future__ import annotations

import pytest

from app.providers.router import provider_key_for_model


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-4o", "openai"),
        ("gpt-4o-mini", "openai"),
        ("o3-mini", "openai"),
        ("text-embedding-3-small", "openai"),
        ("claude-opus-4-8", "anthropic"),
        ("claude-sonnet-4-6", "anthropic"),
        ("llama3.2", "ollama"),
        ("mistral", "ollama"),
        ("qwen2.5", "ollama"),
    ],
)
def test_routing(model, expected):
    assert provider_key_for_model(model) == expected
