"""Logging + deterministic seeding."""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed random / numpy / torch (torch imported lazily)."""
    random.seed(seed)
    np.random.seed(seed)
    import torch  # lazy: lets non-training scripts work without torch installed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_logger(name: str, out_file: Path | None = None,
                level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); logger.addHandler(sh)
    if out_file is not None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(out_file); fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
