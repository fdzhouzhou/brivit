"""Run the Brivit main experiment on TORGO.

Loops over (seed, fold) for the main model, writes per-run + aggregated
CSV/JSON tables directly to a timestamped run directory.

Usage:
    python -m scripts.run_main --config configs/default.yaml

The manifest file referenced by `split.manifest_path` must already exist.
Build it once with:
    python -m scripts.build_split --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from brivit.config.loader import load_config, save_config_snapshot
from brivit.data.splits import SplitManifest, discover_torgo
from brivit.evaluation.aggregate import write_main_results
from brivit.training.train_main import train_one
from brivit.utils.logging_utils import make_logger
from brivit.utils.runs import new_run
from brivit.utils.runtime_meta import write_runtime_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--folds", nargs="*", type=int, default=None,
                    help="Optional subset of fold indices (default: all).")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="Optional subset of seeds (default: cfg).")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    if cfg["experiment"].get("legacy_sample_split", False):
        raise RuntimeError(
            "experiment.legacy_sample_split=true is set. The main flow refuses "
            "to run under the legacy sample-level split (revision #15, #64)."
        )

    # --- paths & logging
    paths = new_run(cfg["experiment"]["output_root"],
                    cfg["experiment"]["name"])
    log = make_logger("brivit.main", paths.logs / "main.log")
    save_config_snapshot(cfg, paths.root)

    # --- manifest
    manifest_path = Path(cfg["split"]["manifest_path"])
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found at {manifest_path}. "
            f"Run: python -m scripts.build_split --config {args.config}"
        )
    manifest = SplitManifest.from_json(manifest_path)
    # mirror a copy into the run dir so results are fully self-contained (#50)
    shutil.copy2(manifest_path, paths.split_manifest)

    # --- samples
    samples = discover_torgo(cfg["dataset"]["root"])
    log.info(f"Discovered {len(samples)} samples across "
             f"{len(manifest.speakers)} speakers.")

    # --- execute
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]["model_init"]
    folds = [f for f in manifest.folds
             if (args.folds is None or f.index in args.folds)]

    rows: list[dict] = []
    for seed in seeds:
        for fold in folds:
            log.info(f"=== Brivit | seed={seed} | fold={fold.index} ===")
            row = train_one(cfg, samples, fold, seed, paths, logger=log)
            rows.append(row)
            # Incremental persistence so a crash doesn't lose everything.
            with open(paths.results / "_partial_main_rows.json", "w") as f:
                json.dump(rows, f, indent=2, default=str)

    # --- final writers
    out = write_main_results(rows, paths.results)
    log.info("main_results_per_run.csv / main_results_summary.csv written.")
    log.info(f"Summary rows:\n{out['summary'].to_string(index=False)}")

    paths.write_meta({
        "experiment": cfg["experiment"]["name"],
        "seeds": seeds,
        "folds": [f.index for f in folds],
        "manifest_path": str(manifest_path),
        "n_rows": len(rows),
        "subject_disjoint": True,
    })
    write_runtime_meta(paths.root, config=cfg,
                       split_manifest_path=manifest_path,
                       component="run_main")


if __name__ == "__main__":
    main()
