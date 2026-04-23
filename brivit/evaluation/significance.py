"""Paired significance tests between the main model and baselines (#39-#40).

Pairing is over matched (seed, fold) runs: both models saw the same data split,
so each (seed, fold) pair is one matched observation of a metric.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _paired_values(df: pd.DataFrame, model_a: str, model_b: str,
                   metric: str) -> tuple[np.ndarray, np.ndarray]:
    a = df[df["model"] == model_a].set_index(["seed", "fold"])[metric]
    b = df[df["model"] == model_b].set_index(["seed", "fold"])[metric]
    common = sorted(set(a.index) & set(b.index))
    if not common:
        return np.array([]), np.array([])
    return a.loc[common].values, b.loc[common].values


def paired_wilcoxon(df: pd.DataFrame, main_model: str,
                    comparators: Iterable[str],
                    metric: str = "roc_auc",
                    correction: str = "holm",
                    alpha: float = 0.05) -> dict:
    """Return {comparator -> {stat, p_raw, p_corrected, n_pairs, metric,
                              median_diff}}."""
    records = []
    for comp in comparators:
        a, b = _paired_values(df, main_model, comp, metric)
        if a.size < 2:
            records.append({
                "comparator": comp, "n_pairs": int(a.size),
                "stat": float("nan"), "p_raw": float("nan"),
                "median_diff": float("nan"),
            })
            continue
        diff = a - b
        try:
            stat, p = wilcoxon(a, b, zero_method="wilcox",
                               alternative="two-sided")
        except ValueError:
            stat, p = float("nan"), 1.0
        records.append({
            "comparator": comp, "n_pairs": int(a.size),
            "stat": float(stat), "p_raw": float(p),
            "median_diff": float(np.median(diff)),
        })
    # Holm-Bonferroni correction
    if correction == "holm":
        ordered = sorted(enumerate(records), key=lambda kv: kv[1]["p_raw"])
        m = len(ordered)
        for rank, (i, rec) in enumerate(ordered):
            p_adj = min(1.0, rec["p_raw"] * (m - rank))
            records[i]["p_corrected"] = p_adj
            records[i]["significant"] = bool(p_adj < alpha)
    else:
        for rec in records:
            rec["p_corrected"] = rec["p_raw"]
            rec["significant"] = bool(rec["p_raw"] < alpha)
    return {
        "main_model": main_model,
        "metric": metric,
        "correction": correction,
        "alpha": alpha,
        "comparisons": records,
    }
