"""Build and export the TORGO subject-disjoint split manifest.

Required FIRST STEP before any experiment is run. All downstream scripts
(main, baselines, ablation) consume this manifest, so they cannot diverge
on which speaker lands in which fold (#14, #29, #32).

Produces, at the manifest's parent directory:
    * split_manifest.json          — speaker lists per fold
    * split_quality.json / .csv    — speaker/sample/label/gender counts
    * also prints a short table to the terminal

Fold validity is enforced: empty split or single-class test fold raises
``FoldQualityError`` (revision #6). Pass ``--no-strict`` to downgrade to
warnings — useful for debugging uneven datasets but never acceptable for
the main paper table.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from brivit.config.loader import load_config
from brivit.data.splits import (build_manifest, build_quality_report,
                                discover_torgo, format_quality_table,
                                write_quality_report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[],
                    help="Dot-path overrides, e.g. split.k_folds=3")
    ap.add_argument("--no-strict", action="store_true",
                    help="Downgrade empty / single-class fold violations to "
                         "warnings. Do not use for paper runs.")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    manifest = build_manifest(cfg)

    # Manifest first (already contains its own disjointness assertions).
    manifest_path = Path(cfg["split"]["manifest_path"])
    manifest.to_json(manifest_path)

    # Quality report alongside.
    samples = discover_torgo(cfg["dataset"]["root"])
    records = build_quality_report(samples, manifest,
                                   strict=not args.no_strict)
    out_dir = manifest_path.parent
    q_json, q_csv = write_quality_report(records, out_dir)

    # Terminal summary
    print(f"[ok] Split protocol: {manifest.protocol}")
    print(f"[ok] Folds:          {manifest.k_folds}")
    print(f"[ok] Speakers:       {len(manifest.speakers)}")
    print(f"[ok] Manifest:       {manifest_path}")
    print(f"[ok] Quality report: {q_json} / {q_csv}")
    print()
    print("Per-fold composition (speakers / samples / test label+gender):")
    print(format_quality_table(records))


if __name__ == "__main__":
    main()
