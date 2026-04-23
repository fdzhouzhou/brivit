"""Aggregate per-(seed, fold) metric rows into paper-ready tables.

Covers:
    (#38) mean, std, 95% CI across seeds.
    (#47) write main_results / baselines / ablation / statistics CSVs + JSON.
    (#49) tables are directly usable — no manual post-processing.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

METRIC_COLS = [
    "accuracy", "precision", "recall", "specificity",
    "f1", "balanced_accuracy", "roc_auc", "pr_auc",
    "brier", "ece",
]


# The summary-level CI is computed from the mean ± 1.96·SEM of (seed, fold)
# point estimates.  It is a NORMAL-APPROXIMATION CI, distinct from the
# per-run BOOTSTRAP CI.  To keep these two layers visually unambiguous in
# downstream CSVs we use distinct column suffixes (revision #5):
#     per-run  : <metric>_ci_lo,            <metric>_ci_hi
#     summary  : <metric>_summary_ci95_lo,  <metric>_summary_ci95_hi
SUMMARY_CI_METHOD = "normal_approximation"
PER_RUN_CI_METHOD = "bootstrap"


def _mean_std_ci(x: np.ndarray) -> dict:
    x = np.asarray([v for v in x if not (isinstance(v, float) and math.isnan(v))],
                   dtype=float)
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "summary_ci95_lo": float("nan"),
                "summary_ci95_hi": float("nan"),
                "n": 0}
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    if x.size > 1:
        sem = std / math.sqrt(x.size)
        lo = mean - 1.96 * sem
        hi = mean + 1.96 * sem
    else:
        lo = hi = mean
    return {"mean": mean, "std": std,
            "summary_ci95_lo": lo, "summary_ci95_hi": hi,
            "n": x.size}


def aggregate_rows(rows: list[dict],
                   group_cols: list[str]) -> pd.DataFrame:
    """Group rows by e.g. ['model'] or ['model','ablation'] and aggregate metrics.

    Each row in `rows` is a per-(seed, fold) dict with METRIC_COLS keys.
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        for col in METRIC_COLS:
            if col not in sub.columns:
                continue
            agg = _mean_std_ci(sub[col].values)
            for k, v in agg.items():
                rec[f"{col}_{k}"] = v
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------- writers
def _write_ci_methods(out_dir: Path) -> None:
    """Emit a small JSON declaring how the two CI layers were computed.

    Revision #5: any summary table must be auditable — readers should be able
    to see, without opening the code, whether a CI column came from a
    per-run bootstrap or a normal-approximation over (seed, fold) means.
    """
    payload = {
        "per_run_ci": {
            "method": PER_RUN_CI_METHOD,      # "bootstrap"
            "columns": ["<metric>_ci_lo", "<metric>_ci_hi"],
            "description": ("Percentile bootstrap (default 1000 resamples) "
                            "over a single test split. Covers aleatoric "
                            "uncertainty inside one (seed, fold)."),
        },
        "summary_ci": {
            "method": SUMMARY_CI_METHOD,      # "normal_approximation"
            "columns": ["<metric>_summary_ci95_lo", "<metric>_summary_ci95_hi"],
            "description": ("Mean ± 1.96·SEM computed over per-(seed, fold) "
                            "point estimates. Covers between-run variability. "
                            "This is NOT a bootstrap CI."),
        },
    }
    with open(out_dir / "ci_methods.json", "w") as f:
        json.dump(payload, f, indent=2)


def write_main_results(rows: list[dict], out_dir: Path) -> dict:
    """Main table for Brivit (#47, #49, #62). Only subject-disjoint rows allowed."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # Guard against accidental leakage of non-subject-disjoint rows
    for r in rows:
        if not r.get("subject_disjoint", False):
            raise AssertionError(
                f"Row not flagged subject_disjoint=True: {r.get('model')}, "
                f"seed={r.get('seed')}, fold={r.get('fold')}. Refusing to write "
                "a non-subject-disjoint main result (#63, #64)."
            )
    per_run = pd.DataFrame(rows)
    per_run.to_csv(out_dir / "main_results_per_run.csv", index=False)
    summary = aggregate_rows(rows, group_cols=["model"])
    summary.to_csv(out_dir / "main_results_summary.csv", index=False)
    with open(out_dir / "main_results_summary.json", "w") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)
    _write_ci_methods(out_dir)
    return {"per_run": per_run, "summary": summary}


def write_baseline_results(rows: list[dict], out_dir: Path) -> pd.DataFrame:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    per_run = pd.DataFrame(rows)
    per_run.to_csv(out_dir / "baselines_per_run.csv", index=False)
    summary = aggregate_rows(rows, group_cols=["model"])
    summary.to_csv(out_dir / "baselines_summary.csv", index=False)
    _write_ci_methods(out_dir)
    return summary


def write_ablation_results(rows: list[dict], out_dir: Path) -> pd.DataFrame:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    per_run = pd.DataFrame(rows)
    per_run.to_csv(out_dir / "ablation_per_run.csv", index=False)
    summary = aggregate_rows(rows, group_cols=["ablation_name"])
    summary.to_csv(out_dir / "ablation_summary.csv", index=False)
    _write_ci_methods(out_dir)
    return summary


def write_statistics(results: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "statistics.json", "w") as f:
        json.dump(results, f, indent=2)
    # Also flatten to CSV for spreadsheet users
    flat = []
    for comparison, payload in results.items():
        row = {"comparison": comparison}
        row.update(payload)
        flat.append(row)
    pd.DataFrame(flat).to_csv(out_dir / "statistics.csv", index=False)
