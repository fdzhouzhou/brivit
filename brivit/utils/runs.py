"""Run-directory management: timestamped folders, config snapshot,
split manifest, checkpoints, result summaries — per requirement #50.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class RunPaths:
    root: Path                # runs/<experiment>/<timestamp>/
    config_snapshot: Path     # <root>/config.snapshot.yaml
    split_manifest: Path      # <root>/split_manifest.json
    checkpoints: Path         # <root>/checkpoints/
    logs: Path                # <root>/logs/
    results: Path             # <root>/results/
    meta: Path                # <root>/meta.json

    def create(self) -> None:
        for p in (self.root, self.checkpoints, self.logs, self.results):
            p.mkdir(parents=True, exist_ok=True)

    def write_meta(self, data: dict) -> None:
        with open(self.meta, "w") as f:
            json.dump(data, f, indent=2, default=str)


def new_run(output_root: str, experiment_name: str) -> RunPaths:
    ts = time.strftime("%Y%m%d_%H%M%S")
    root = Path(output_root) / experiment_name / ts
    paths = RunPaths(
        root=root,
        config_snapshot=root / "config.snapshot.yaml",
        split_manifest=root / "split_manifest.json",
        checkpoints=root / "checkpoints",
        logs=root / "logs",
        results=root / "results",
        meta=root / "meta.json",
    )
    paths.create()
    return paths
