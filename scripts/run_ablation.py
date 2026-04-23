"""Ablation runner.

Implements the six switches mandated by revision #32:
    use_augment            — augmentation on/off
    use_spectral           — Mel/MFCC spectral branch
    use_spike              — LIF spike branch + post-processing
    use_fusion             — spectral⊕spike fusion
    use_dual_channel_attn  — dual-channel attention vs plain MHA
    use_gender_branch      — unified model (False, default) vs per-gender

Each ablation config differs from the default by ONE switch only (#33).
Results are written to a single ablation table (#34).

Usage:
    python -m scripts.run_ablation --config configs/default.yaml
    python -m scripts.run_ablation --only use_dual_channel_attn use_augment
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

from brivit.config.loader import load_config, save_config_snapshot
from brivit.data.splits import SplitManifest, discover_torgo, samples_for_fold
from brivit.evaluation.aggregate import write_ablation_results
from brivit.evaluation.metrics import evaluate_arrays
from brivit.training.train_main import train_one, train_one_return_preds
from brivit.utils.logging_utils import make_logger
from brivit.utils.runs import new_run
from brivit.utils.runtime_meta import write_runtime_meta

# All six switches. "baseline_default" is the default config (for reference).
ABLATION_SWITCHES = [
    "use_augment", "use_spectral", "use_spike",
    "use_fusion", "use_dual_channel_attn", "use_gender_branch",
]


def _cfg_with_switch_off(base_cfg: dict, switch: str) -> dict:
    """Return a deep-copied config with exactly one switch flipped OFF
    (or ON, for use_gender_branch, which defaults to False)."""
    cfg = copy.deepcopy(base_cfg)
    cfg["ablation"] = cfg.get("ablation", {})
    if switch == "use_gender_branch":
        # default is False -> flip ON to test the gender-aware variant
        cfg["ablation"][switch] = True
    else:
        cfg["ablation"][switch] = False
    # A couple of consistency patches so internal modules don't fight:
    if switch == "use_augment":
        cfg["augmentation"]["enabled"] = False
    if switch in {"use_spectral", "use_spike"} \
            and not cfg["ablation"].get("use_spectral", True) \
            and not cfg["ablation"].get("use_spike", True):
        # forbid the invalid combo (nothing left to feed the model)
        raise ValueError("Cannot disable BOTH spectral and spike branches.")
    return cfg


def _gender_filter(samples: list[dict], gender: str) -> list[dict]:
    return [s for s in samples if s["gender"] == gender]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None,
                    help="Subset of switches to test (default: all six).")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    base_cfg = load_config(args.config, overrides=args.override)
    paths = new_run(base_cfg["experiment"]["output_root"],
                    base_cfg["experiment"]["name"] + "__ablation")
    log = make_logger("brivit.ablation", paths.logs / "ablation.log")
    save_config_snapshot(base_cfg, paths.root)

    manifest = SplitManifest.from_json(base_cfg["split"]["manifest_path"])
    shutil.copy2(base_cfg["split"]["manifest_path"], paths.split_manifest)
    samples = discover_torgo(base_cfg["dataset"]["root"])

    seeds = args.seeds if args.seeds is not None else base_cfg["seeds"]["model_init"]
    folds = [f for f in manifest.folds
             if (args.folds is None or f.index in args.folds)]
    switches = args.only if args.only is not None else ABLATION_SWITCHES
    unknown = [s for s in switches if s not in ABLATION_SWITCHES]
    if unknown:
        raise KeyError(f"Unknown switches: {unknown}. "
                       f"Available: {ABLATION_SWITCHES}")

    rows: list[dict] = []

    # --- (1) Reference row: full model with every switch ON
    log.info("=== ablation: full_model (all switches ON) ===")
    for seed in seeds:
        for fold in folds:
            r = train_one(base_cfg, samples, fold, seed, paths, logger=log)
            r["ablation_name"] = "full_model"
            r["switch_flipped"] = "none"
            rows.append(r)

    # --- (2) For each switch: flip ONE off, keep others at default
    for sw in switches:
        try:
            cfg_i = _cfg_with_switch_off(base_cfg, sw)
        except ValueError as e:
            log.warning(f"skipping {sw}: {e}")
            continue

        if sw == "use_gender_branch":
            # Revision #2: the "gender-aware" variant is ONE system — two
            # branches trained independently on each gender, their test
            # predictions concatenated and scored ONCE for a single unified
            # row per (seed, fold). We do NOT emit per-gender rows here.
            import numpy as np
            log.info("=== ablation: gender_aware_branch (unified) ===")
            f_samples = _gender_filter(samples, "F")
            m_samples = _gender_filter(samples, "M")
            for seed in seeds:
                for fold in folds:
                    try:
                        yt_f, yp_f, ys_f = train_one_return_preds(
                            cfg_i, f_samples, fold, seed, paths, logger=log)
                        yt_m, yp_m, ys_m = train_one_return_preds(
                            cfg_i, m_samples, fold, seed, paths, logger=log)
                    except Exception as e:
                        # A gender may have no speakers in a fold's test set
                        # (small dataset). Log and skip that fold.
                        log.warning(f"gender-aware fold {fold.index} seed {seed}"
                                    f" skipped: {e}")
                        continue
                    y_true = np.concatenate([yt_f, yt_m])
                    y_pred = np.concatenate([yp_f, yp_m])
                    y_score = np.concatenate([ys_f, ys_m])
                    r = evaluate_arrays(
                        y_true, y_pred, y_score,
                        n_bins=base_cfg["evaluation"]["ece_bins"],
                        bootstrap_iters=base_cfg["evaluation"]["bootstrap_iters"],
                        seed=seed,
                    )
                    r.update({
                        "model": "Brivit",
                        "seed": seed,
                        "fold": fold.index,
                        "subject_disjoint": True,
                        "ablation_name": "gender_aware_branch",
                        "switch_flipped": sw,
                        # Record the sub-system composition for audit.
                        "n_female_test": int(len(yt_f)),
                        "n_male_test":   int(len(yt_m)),
                    })
                    rows.append(r)
        else:
            log.info(f"=== ablation: no_{sw} ===")
            for seed in seeds:
                for fold in folds:
                    r = train_one(cfg_i, samples, fold, seed, paths, logger=log)
                    r["ablation_name"] = f"no_{sw}"
                    r["switch_flipped"] = sw
                    rows.append(r)

        # incremental snapshot
        with open(paths.results / "_partial_ablation_rows.json", "w") as f:
            json.dump(rows, f, indent=2, default=str)

    summary = write_ablation_results(rows, paths.results)
    log.info("ablation_per_run.csv / ablation_summary.csv written.")
    log.info(f"\n{summary.to_string(index=False)}")

    paths.write_meta({
        "experiment": base_cfg["experiment"]["name"] + "__ablation",
        "switches": switches,
        "seeds": seeds,
        "folds": [f.index for f in folds],
        "n_rows": len(rows),
    })
    write_runtime_meta(paths.root, config=base_cfg,
                       split_manifest_path=base_cfg["split"]["manifest_path"],
                       component="run_ablation")


if __name__ == "__main__":
    main()
