"""Run a baseline model for one (seed, fold) combination."""
from __future__ import annotations

from brivit.baselines.runners import BASELINE_REGISTRY
from brivit.data.splits import samples_for_fold
from brivit.utils.logging_utils import make_logger, set_seed


def train_baseline_one(name: str, cfg: dict, samples, fold,
                       seed: int, run_paths, logger=None) -> dict:
    logger = logger or make_logger(
        "brivit.baseline", run_paths.logs / "baselines.log"
    )
    set_seed(seed, cfg["seeds"]["deterministic"])
    if name not in BASELINE_REGISTRY:
        raise KeyError(f"Unknown baseline: {name}")
    sbs = samples_for_fold(samples, fold)
    logger.info(f"[baseline={name} fold={fold.index} seed={seed}] starting")
    row = BASELINE_REGISTRY[name](sbs, cfg, seed)
    row.update({"seed": seed, "fold": fold.index, "subject_disjoint": True})
    return row
