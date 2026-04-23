"""Tests for metric CI coverage and the unified bootstrap loop.

Pure-Python — marked `unit`, runs in the CPU-only test set.
"""
from __future__ import annotations

import numpy as np
import pytest

from brivit.evaluation.metrics import (CORE_CI_METRICS, bootstrap_ci_all,
                                       evaluate_arrays)

pytestmark = pytest.mark.unit


def _synthetic():
    rng = np.random.default_rng(42)
    N = 200
    y_true = rng.integers(0, 2, N)
    y_score = np.clip(y_true + rng.normal(0, 0.3, N), 0, 1)
    y_pred = (y_score >= 0.5).astype(int)
    return y_true, y_pred, y_score


def test_every_core_metric_has_ci_columns():
    """Revision item 3: all core metrics must have CI columns."""
    y_true, y_pred, y_score = _synthetic()
    row = evaluate_arrays(y_true, y_pred, y_score, bootstrap_iters=200)
    for m in CORE_CI_METRICS:
        assert f"{m}_ci_lo" in row, f"missing {m}_ci_lo"
        assert f"{m}_ci_hi" in row, f"missing {m}_ci_hi"
        # CI bounds must bracket the point estimate
        pe = row[m]
        lo = row[f"{m}_ci_lo"]
        hi = row[f"{m}_ci_hi"]
        assert lo <= pe <= hi, f"{m}: pe={pe} not in [{lo}, {hi}]"


def test_bootstrap_ci_all_returns_all_metrics_in_one_call():
    """Revision item 3: bootstrap logic unified — one call, all metrics."""
    y_true, y_pred, y_score = _synthetic()
    ci = bootstrap_ci_all(y_true, y_pred, y_score, n_boot=100)
    assert set(ci) == set(CORE_CI_METRICS)
    for m, (lo, hi) in ci.items():
        assert 0.0 <= lo <= hi <= 1.0 or np.isnan(lo), \
            f"{m} CI looks wrong: {lo}, {hi}"


def test_zero_bootstrap_skips_ci():
    """bootstrap_iters=0 should produce point estimates only, no CIs."""
    y_true, y_pred, y_score = _synthetic()
    row = evaluate_arrays(y_true, y_pred, y_score, bootstrap_iters=0)
    for m in CORE_CI_METRICS:
        assert f"{m}_ci_lo" not in row
        assert f"{m}_ci_hi" not in row
        assert m in row


def test_calibration_columns_present():
    y_true, y_pred, y_score = _synthetic()
    row = evaluate_arrays(y_true, y_pred, y_score, bootstrap_iters=50)
    assert "brier" in row and "ece" in row


def test_every_metric_also_has_point_estimate():
    """Per-run row must expose both a point-estimate and CI for every
    core metric (revision #9)."""
    y_true, y_pred, y_score = _synthetic()
    row = evaluate_arrays(y_true, y_pred, y_score, bootstrap_iters=50)
    for m in CORE_CI_METRICS:
        assert m in row, f"missing point estimate: {m}"
        assert f"{m}_ci_lo" in row, f"missing CI lower: {m}"
        assert f"{m}_ci_hi" in row, f"missing CI upper: {m}"
    assert {"tp", "fp", "tn", "fn"} <= set(row), "confusion matrix missing"
