"""Aggregator tests: rejects non-subject-disjoint rows (#63, #64) and
produces mean/std/CI columns for every core metric (#42, #44).

Pure-Python — marked `unit`, runs in the CPU-only test set.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from brivit.evaluation.aggregate import aggregate_rows, write_main_results
from brivit.evaluation.metrics import CORE_CI_METRICS

pytestmark = pytest.mark.unit


def _row(model: str, seed: int, fold: int, sd: bool = True) -> dict:
    return {
        "model": model, "seed": seed, "fold": fold,
        "subject_disjoint": sd,
        "accuracy": 0.9, "precision": 0.9, "recall": 0.9,
        "specificity": 0.9, "f1": 0.9, "balanced_accuracy": 0.9,
        "roc_auc": 0.95, "pr_auc": 0.93,
        "brier": 0.05, "ece": 0.03,
    }


def test_main_results_writer_accepts_valid_rows():
    rows = [_row("Brivit", s, f) for s in range(2) for f in range(3)]
    with tempfile.TemporaryDirectory() as td:
        out = write_main_results(rows, Path(td))
        assert (Path(td) / "main_results_per_run.csv").is_file()
        assert (Path(td) / "main_results_summary.csv").is_file()
        assert len(out["summary"]) == 1   # one group: Brivit


def test_main_results_writer_rejects_non_disjoint_row():
    rows = [_row("Brivit", 0, 0, sd=True), _row("Brivit", 0, 1, sd=False)]
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(AssertionError):
            write_main_results(rows, Path(td))


def test_summary_has_mean_std_ci_for_every_core_metric(tmp_path):
    """Summary must expose mean / std / summary_ci95_lo / summary_ci95_hi / n
    for every core metric (revision #3, #5)."""
    rows = []
    for seed in range(5):
        for fold in range(5):
            r = _row("Brivit", seed, fold)
            # vary slightly so std > 0
            for m in CORE_CI_METRICS:
                r[m] = 0.9 + 0.01 * seed - 0.002 * fold
            rows.append(r)
    out = write_main_results(rows, tmp_path)
    summary = out["summary"]
    for m in CORE_CI_METRICS:
        for suffix in ("_mean", "_std", "_summary_ci95_lo",
                       "_summary_ci95_hi", "_n"):
            assert f"{m}{suffix}" in summary.columns, \
                f"summary missing {m}{suffix}"
    # sanity: values are plausible
    row = summary.iloc[0]
    for m in CORE_CI_METRICS:
        assert row[f"{m}_n"] == 25
        lo = row[f"{m}_summary_ci95_lo"]
        hi = row[f"{m}_summary_ci95_hi"]
        assert lo <= row[f"{m}_mean"] <= hi


def test_aggregate_rows_also_exposes_ci_columns():
    rows = [_row("Brivit", s, f) for s in range(3) for f in range(3)]
    df = aggregate_rows(rows, group_cols=["model"])
    for m in CORE_CI_METRICS:
        assert f"{m}_mean" in df.columns
        assert f"{m}_summary_ci95_lo" in df.columns
        assert f"{m}_summary_ci95_hi" in df.columns


def test_ci_methods_json_declares_both_layers(tmp_path):
    """A ci_methods.json sidecar documents how each CI layer was computed."""
    import json
    rows = [_row("Brivit", 0, i) for i in range(3)]
    write_main_results(rows, tmp_path)
    p = tmp_path / "ci_methods.json"
    assert p.is_file()
    payload = json.loads(p.read_text())
    assert payload["per_run_ci"]["method"] == "bootstrap"
    assert payload["summary_ci"]["method"] == "normal_approximation"
