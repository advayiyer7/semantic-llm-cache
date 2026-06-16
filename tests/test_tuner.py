"""Threshold tuner sweep — pure unit tests."""

from __future__ import annotations

from app.cache.tuner import cosine, sweep


def test_cosine_basics():
    assert cosine([1, 0], [1, 0]) == 1.0
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([0, 0], [1, 1]) == 0.0  # zero vector guarded


def test_sweep_metrics():
    # Two should-hit pairs at high similarity, one should-miss at high similarity
    # (a false-positive trap), one should-miss at low similarity.
    scored = [(0.99, True), (0.96, True), (0.97, False), (0.50, False)]

    rows = {r["threshold"]: r for r in sweep(scored, [0.95, 0.98])}

    # At 0.95: 0.99,0.96,0.97 served. tp=2, fp=1 -> precision 2/3, recall 2/2.
    low = rows[0.95]
    assert (low["tp"], low["fp"], low["fn"], low["tn"]) == (2, 1, 0, 1)
    assert low["precision"] == round(2 / 3, 3)
    assert low["recall"] == 1.0
    assert low["hit_rate"] == 0.75

    # At 0.98: only 0.99 served. tp=1, fp=0, the 0.96 should-hit now missed.
    high = rows[0.98]
    assert (high["tp"], high["fp"], high["fn"], high["tn"]) == (1, 0, 1, 2)
    assert high["precision"] == 1.0
    assert high["recall"] == 0.5
