"""Run baselines on the same TORGO subject-disjoint split (#29).

Outputs a unified baselines table with the same metric columns as the main
results (#30), consumable by the summarizer and the statistics runner.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from brivit.baselines.runners import BASELINE_REGISTRY
from brivit.config.loader import load_config, save_config_snapshot
from brivit.data.splits import SplitManifest, discover_torgo
from brivit.evaluation.aggregate import write_baseline_results
from brivit.training.train_baseline import train_baseline_one
from brivit.utils.logging_utils import make_logger
from brivit.utils.runs import new_run
from brivit.utils.runtime_meta import write_runtime_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of baseline names to run (default: "
                         "baselines.enabled from the config).")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    paths = new_run(cfg["experiment"]["output_root"],
                    cfg["experiment"]["name"] + "__baselines")
    log = make_logger("brivit.baselines", paths.logs / "baselines.log")
    save_config_snapshot(cfg, paths.root)

    manifest_path = Path(cfg["split"]["manifest_path"])
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found at {manifest_path}. Build it first "
            "with: python -m scripts.build_split"
        )
    manifest = SplitManifest.from_json(manifest_path)
    shutil.copy2(manifest_path, paths.split_manifest)
    samples = discover_torgo(cfg["dataset"]["root"])

    models = args.models if args.models is not None else cfg["baselines"]["enabled"]
    unknown = [m for m in models if m not in BASELINE_REGISTRY]
    if unknown:
        raise KeyError(f"Unknown baselines: {unknown}. "
                       f"Available: {list(BASELINE_REGISTRY)}")

    seeds = args.seeds if args.seeds is not None else cfg["seeds"]["model_init"]
    folds = [f for f in manifest.folds
             if (args.folds is None or f.index in args.folds)]

    rows: list[dict] = []
    for m in models:
        for seed in seeds:
            for fold in folds:
                log.info(f"=== {m} | seed={seed} | fold={fold.index} ===")
                row = train_baseline_one(m, cfg, samples, fold, seed, paths,
                                         logger=log)
                rows.append(row)
                with open(paths.results / "_partial_baseline_rows.json", "w") as f:
                    json.dump(rows, f, indent=2, default=str)

    summary = write_baseline_results(rows, paths.results)
    log.info("baselines_per_run.csv / baselines_summary.csv written.")
    log.info(f"\n{summary.to_string(index=False)}")

    paths.write_meta({
        "experiment": cfg["experiment"]["name"] + "__baselines",
        "models": models,
        "seeds": seeds,
        "folds": [f.index for f in folds],
        "manifest_path": str(manifest_path),
        "n_rows": len(rows),
        "subject_disjoint": True,
    })
    write_runtime_meta(paths.root, config=cfg,
                       split_manifest_path=manifest_path,
                       component="run_baselines")


if __name__ == "__main__":
    main()
