"""Config loading with optional CLI overrides.

Usage:
    from brivit.config.loader import load_config
    cfg = load_config("configs/default.yaml", overrides=["train.epochs=50"])
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


def _set_path(d: dict, dotted: str, value: Any) -> None:
    """Set cfg[a][b][c] = value given dotted string 'a.b.c'."""
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    # Try to keep types consistent with YAML
    try:
        parsed = yaml.safe_load(value) if isinstance(value, str) else value
    except Exception:
        parsed = value
    cur[keys[-1]] = parsed


def load_config(path: str | Path,
                overrides: Iterable[str] | None = None) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for o in overrides:
            if "=" not in o:
                raise ValueError(f"Bad override (expected key=value): {o!r}")
            k, v = o.split("=", 1)
            _set_path(cfg, k.strip(), v.strip())
    return cfg


def save_config_snapshot(cfg: Mapping, out_dir: str | Path) -> Path:
    """Persist the effective config next to the run outputs (#50)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "config.snapshot.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False)
    return out


def deep_merge(base: Mapping, override: Mapping) -> dict:
    """Deep-merge two mappings, override wins on leaves."""
    out = copy.deepcopy(dict(base))
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
