# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Tests for SavingsEstimator (optimize/metrics/estimator.py)."""

import pytest

from superlocalmemory.optimize.metrics.estimator import SavingsEstimator
from superlocalmemory.optimize.storage.db import MetricsSnapshot


def test_estimate_basic():
    est = SavingsEstimator()
    snap = MetricsSnapshot(
        tokens_saved_input=1000,
        tokens_saved_output=500,
        tokens_saved_compress=200,
    )
    result = est.estimate(snap, provider="anthropic")
    assert result["usd"] > 0
    assert result["inr"] > 0
    assert result["tokens_saved_total"] == 1700
    assert result["cache_tokens"] == 1500
    assert result["compress_tokens"] == 200
    assert result["pricing_date"] == "2026-06-07"


def test_estimate_zero():
    est = SavingsEstimator()
    snap = MetricsSnapshot()
    result = est.estimate(snap, provider="anthropic")
    assert result["usd"] == 0.0
    assert result["inr"] == 0.0
    assert result["tokens_saved_total"] == 0


def test_estimate_openai():
    est = SavingsEstimator()
    snap = MetricsSnapshot(tokens_saved_input=1_000_000)
    result = est.estimate(snap, provider="openai")
    assert result["usd"] == 2.50


def test_estimate_gemini():
    est = SavingsEstimator()
    snap = MetricsSnapshot(tokens_saved_input=1_000_000)
    result = est.estimate(snap, provider="gemini")
    assert result["usd"] == 1.25


def test_estimate_unknown_provider_falls_back():
    est = SavingsEstimator()
    snap = MetricsSnapshot(tokens_saved_input=1_000_000)
    result = est.estimate(snap, provider="unknown_provider")
    assert result["usd"] == 3.00  # anthropic fallback


def _today_is(monkeypatch, day):
    """Pin the estimator's clock: freshness is a property of a date, not of
    whenever the suite happens to run."""
    import datetime as _dt

    from superlocalmemory.optimize.metrics import estimator as mod

    class _Date(_dt.date):
        @classmethod
        def today(cls):
            return day

    monkeypatch.setattr(mod, "date", _Date)


def test_is_stale_is_false_while_the_pricing_table_is_fresh(monkeypatch):
    from datetime import date

    from superlocalmemory.optimize.metrics import estimator as mod

    est = SavingsEstimator()
    _today_is(monkeypatch, date.fromisoformat(est._PRICING_DATE))
    assert est._is_stale() is False
    _today_is(monkeypatch, date.fromisoformat(est._PRICING_DATE)
              + mod.timedelta(days=mod._PRICING_STALE_DAYS))
    assert est._is_stale() is False, "the last fresh day is still fresh"


def test_is_stale_is_true_once_the_pricing_table_ages_out(monkeypatch):
    from datetime import date

    from superlocalmemory.optimize.metrics import estimator as mod

    est = SavingsEstimator()
    _today_is(monkeypatch, date.fromisoformat(est._PRICING_DATE)
              + mod.timedelta(days=mod._PRICING_STALE_DAYS + 1))
    assert est._is_stale() is True
