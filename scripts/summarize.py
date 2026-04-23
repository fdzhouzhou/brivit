"""Summarize main + baseline results into a single paper-ready table
and run paired Wilcoxon significance tests between Brivit and each baseline.

Usage:
    python -m scripts.summarize \\
        --main_dir  runs/brivit_torgo_subject_disjoint/<ts>/results \\
        --base_dir  runs/brivit_torgo_subject_disjoint__baselines/<ts>/results \\
        --out_dir   runs/summary

The "main" directory is the one produced by `scripts/run_main.py`; the "base"
directory is the one produced by `scripts/run_baselines.py`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from brivit.evaluation.aggregate import (aggregate_rows, write_main_results,
                                         write_statistics)
from brivit.evaluation.significance import paired_wilcoxon
from brivit.utils.runtime_meta import write_runtime_meta


def _load_per_run(main_dir: Path, base_dir: Path | None) -> pd.DataFrame:
    frames = []
    mp = main_dir / "main_results_per_run.csv"
    if not mp.is_file():
        raise FileNotFoundError(f"Missing main per-run CSV: {mp}")
    frames.append(pd.read_csv(mp))
    if base_dir is not None:
        bp = base_dir / "baselines_per_run.csv"
        if bp.is_file():
            frames.append(pd.read_csv(bp))
        else:
            print(f"[warn] no baselines_per_run.csv at {bp}; skipping baselines.")
    df = pd.concat(frames, ignore_index=True)
    # (#48, #62-63) Only subject-disjoint rows may contribute to the summary.
    if "subject_disjoint" not in df.columns:
        raise RuntimeError(
            "per-run tables must have a 'subject_disjoint' column."
        )
    leaked = df[df["subject_disjoint"] != True]
    if len(leaked):
        raise RuntimeError(
            f"Refusing to summarize {len(leaked)} non-subject-disjoint rows; "
            "these would violate the main-results contract (#63, #64)."
        )
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_dir", required=True,
                    help="results/ dir from scripts/run_main.py")
    ap.add_argument("--base_dir", default=None,
                    help="results/ dir from scripts/run_baselines.py (optional)")
    ap.add_argument("--out_dir", default="runs/summary")
    ap.add_argument("--metric", default="roc_auc",
                    help="Metric used in paired significance tests.")
    ap.add_argument("--main_name", default="Brivit",
                    help="Row label for the main model in the combined table.")
    args = ap.parse_args()

    main_dir = Path(args.main_dir)
    base_dir = Path(args.base_dir) if args.base_dir else None
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_per_run(main_dir, base_dir)

    # --- Paper table: one row per model, mean/std/CI for each metric
    summary = aggregate_rows(df.to_dict(orient="records"), group_cols=["model"])
    summary.to_csv(out_dir / "paper_main_table.csv", index=False)
    with open(out_dir / "paper_main_table.json", "w") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)

    # --- Statistics: Brivit vs every other model
    other = sorted(m for m in df["model"].unique() if m != args.main_name)
    if not other:
        print(f"[warn] no comparators besides {args.main_name}; skipping tests.")
        return
    stats = paired_wilcoxon(df, args.main_name, other,
                            metric=args.metric, correction="holm")
    write_statistics({f"{args.main_name}_vs_baselines": stats}, out_dir)
    write_runtime_meta(out_dir, component="summarize")

    print(f"[ok] Wrote summary to {out_dir}")
    print(summary.to_string(index=False))
    print()
    print(f"Paired Wilcoxon ({args.main_name} vs others) on '{args.metric}':")
    for rec in stats["comparisons"]:
        print(f"  vs {rec['comparator']:15s}  "
              f"n={rec['n_pairs']:2d}  "
              f"median_diff={rec['median_diff']:+.4f}  "
              f"p_raw={rec['p_raw']:.4f}  "
              f"p_holm={rec['p_corrected']:.4f}  "
              f"sig={rec['significant']}")


if __name__ == "__main__":
    main()
