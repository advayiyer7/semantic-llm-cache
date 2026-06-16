"""Workload generator — pure unit tests."""

from __future__ import annotations

from loadtest.workload import build_workload, mix_counts


def test_workload_is_deterministic():
    a = build_workload(500, seed=7)
    b = build_workload(500, seed=7)
    assert a == b


def test_workload_mix_is_roughly_as_requested():
    items = build_workload(2000, seed=7, mix=(0.4, 0.3, 0.3))
    counts = mix_counts(items)
    assert sum(counts.values()) == 2000
    # Early iterations are forced unique (no history yet), so allow a margin.
    assert 0.30 <= counts["exact"] / 2000 <= 0.45
    assert 0.25 <= counts["paraphrase"] / 2000 <= 0.35
    assert counts["unique"] >= 1


def test_first_item_is_always_unique():
    items = build_workload(10, seed=1)
    assert items[0]["kind"] == "unique"
