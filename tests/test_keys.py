"""Namespace isolation — pure unit tests, no Redis required."""

from __future__ import annotations

from app.cache.keys import cache_namespace


def test_same_inputs_produce_same_namespace():
    a = cache_namespace("gpt-4o", "sys", {"temperature": 0.0, "top_p": 1.0})
    b = cache_namespace("gpt-4o", "sys", {"temperature": 0.0, "top_p": 1.0})
    assert a == b


def test_model_change_changes_namespace():
    a = cache_namespace("gpt-4o", "sys", {"temperature": 0.0})
    b = cache_namespace("llama3", "sys", {"temperature": 0.0})
    assert a != b


def test_temperature_change_changes_namespace():
    a = cache_namespace("gpt-4o", "sys", {"temperature": 0.0})
    b = cache_namespace("gpt-4o", "sys", {"temperature": 0.7})
    assert a != b


def test_system_prompt_change_changes_namespace():
    a = cache_namespace("gpt-4o", "be terse", {})
    b = cache_namespace("gpt-4o", "be verbose", {})
    assert a != b


def test_missing_params_are_handled():
    assert cache_namespace("gpt-4o", None, None)
