"""Metrics, confidence intervals, calibration (requirements #41-#46).

Design notes (post-revision):
    * `core_metrics` returns all point-estimates in one pass.
    * `bootstrap_ci_all` runs ONE shared bootstrap loop and derives CIs for
      every core metric from the same resamples — no per-metric loops
      (revision: "bootstrap 逻辑统一").
    * `evaluate_arrays` is the single entry point that writes per-run rows;
      its output schema is identical across the main model and every baseline
      so downstream tables don't diverge.

Per-metric columns emitted for a single run:
    <metric>             — point estimate
    <metric>_ci_lo       — bootstrap lower bound (default 95 %)
    <metric>_ci_hi       — bootstrap upper bound
    brier, ece           — calibration
    tp, fp, tn, fn       — confusion matrix

And when aggregated across (seed, fold) runs by brivit.evaluation.aggregate:
    <metric>_mean, <metric>_std, <metric>_ci95_lo, <metric>_ci95_hi, <metric>_n
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

# --------------------------------------------------------------------- metrics
# Metrics that get a bootstrap CI by default. `precision` is included because
# it appears in the main paper table (revision item 1).
CORE_CI_METRICS: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",                # == sensitivity
    "specificity",
    "f1",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
)


def core_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                 y_score: np.ndarray) -> Dict[str, float]:
    """Point-estimate of every core metric + confusion matrix counts."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    out: Dict[str, float] = {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "precision":         float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":            float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity":       float(spec),
        "f1":                float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if len(set(y_true.tolist())) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    out["tp"], out["fp"], out["tn"], out["fn"] = (int(tp), int(fp),
                                                   int(tn), int(fn))
    return out


# ------------------------------------------------------------- calibration
def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_score, edges, right=True).clip(1, n_bins) - 1
    N = len(y_score)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = y_score[m].mean()
        acc = (y_true[m] == (y_score[m] >= 0.5).astype(int)).mean()
        ece += m.sum() / N * abs(acc - conf)
    return float(ece)


def calibration_metrics(y_true, y_score, n_bins: int = 10) -> Dict[str, float]:
    return {
        "brier": float(brier_score_loss(y_true, y_score)),
        "ece":   expected_calibration_error(y_true, y_score, n_bins),
    }


# ------------------------------------------------------------- shared bootstrap
def bootstrap_ci_all(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray,
                     metrics: Iterable[str] = CORE_CI_METRICS,
                     n_boot: int = 1000, alpha: float = 0.05,
                     seed: int = 0) -> Dict[str, tuple[float, float]]:
    """One bootstrap loop → CI for every requested metric.

    Replaces the old per-metric resampling pattern (revision: "bootstrap 逻辑统一").
    Returns {metric_name -> (ci_lo, ci_hi)}.
    """
    metrics = tuple(metrics)
    rng = np.random.default_rng(seed)
    N = len(y_true)
    store: dict[str, list[float]] = {m: [] for m in metrics}

    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        yt = y_true[idx]; yp = y_pred[idx]; ys = y_score[idx]
        try:
            vals = core_metrics(yt, yp, ys)
        except Exception:
            continue
        for m in metrics:
            v = vals.get(m, float("nan"))
            if not (isinstance(v, float) and np.isnan(v)):
                store[m].append(v)

    out: dict[str, tuple[float, float]] = {}
    for m, arr in store.items():
        if not arr:
            out[m] = (float("nan"), float("nan"))
            continue
        a = np.asarray(arr, dtype=float)
        out[m] = (float(np.quantile(a, alpha / 2)),
                  float(np.quantile(a, 1 - alpha / 2)))
    return out


# Convenience one-metric wrapper (kept for API compatibility with older callers).
def bootstrap_ci(y_true, y_pred, y_score, metric: str,
                 n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    return bootstrap_ci_all(y_true, y_pred, y_score, metrics=(metric,),
                            n_boot=n_boot, alpha=alpha, seed=seed)[metric]


# ------------------------------------------------------------- aggregated entry
def evaluate_arrays(y_true, y_pred, y_score,
                    n_bins: int = 10,
                    bootstrap_iters: int = 1000,
                    ci_metric_list: Sequence[str] = CORE_CI_METRICS,
                    alpha: float = 0.05,
                    seed: int = 0) -> Dict[str, float]:
    """Point metrics + calibration + bootstrap CIs for all core metrics.

    Every metric in ``CORE_CI_METRICS`` receives ``<m>_ci_lo`` / ``<m>_ci_hi``
    columns.  The summary aggregator additionally exposes across-run
    mean / std / 95 % CI on the point estimates, so the combined summary table
    contains, for each metric, four columns:
        <m>_mean, <m>_std, <m>_ci95_lo, <m>_ci95_hi.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    out = core_metrics(y_true, y_pred, y_score)
    out.update(calibration_metrics(y_true, y_score, n_bins))
    if bootstrap_iters > 0 and ci_metric_list:
        ci = bootstrap_ci_all(y_true, y_pred, y_score,
                              metrics=ci_metric_list,
                              n_boot=bootstrap_iters, alpha=alpha, seed=seed)
        for m, (lo, hi) in ci.items():
            out[f"{m}_ci_lo"] = lo
            out[f"{m}_ci_hi"] = hi
    return out
