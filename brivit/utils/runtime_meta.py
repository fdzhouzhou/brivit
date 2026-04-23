"""Per-run environment / provenance snapshot (revision #10).

Every main/baseline/ablation/summary script emits a ``runtime_meta.json``
into its run directory so that results can be audited later without
consulting logs.

The snapshot covers:
    - Python version, platform, current working directory
    - Installed versions of torch, torchaudio, timm (graceful if absent)
    - CUDA availability, device count, current device name
    - Git commit (if the tree is a git checkout) and the package version
    - Timestamp
    - Hashes of the effective config and the split manifest
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping


def _safe_version(module_name: str) -> str | None:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _hash_mapping(m: Mapping) -> str:
    blob = json.dumps(m, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cuda_info() -> dict:
    info: dict = {"available": False, "device_count": 0,
                  "current_device_name": None}
    try:
        import torch  # local: meta is cheap, torch may or may not exist
        info["available"] = bool(torch.cuda.is_available())
        info["device_count"] = int(torch.cuda.device_count())
        if info["available"] and info["device_count"] > 0:
            info["current_device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def collect_runtime_meta(*, config: Mapping | None = None,
                         split_manifest_path: Path | str | None = None,
                         component: str = "run") -> dict:
    """Assemble a runtime meta payload.  Pure; never touches disk here."""
    here = Path(__file__).resolve().parent
    repo_root = here.parents[1]
    try:
        from brivit import __version__ as pkg_version
    except Exception:
        pkg_version = "unknown"

    payload = {
        "component": component,
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S%z") or time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python":          sys.version.split()[0],
        "platform":        platform.platform(),
        "os":              platform.system(),
        "hostname":        socket.gethostname(),
        "cwd":             os.getcwd(),
        "package_version": pkg_version,
        "git_commit":      _git_commit(repo_root),
        "versions": {
            "torch":       _safe_version("torch"),
            "torchaudio":  _safe_version("torchaudio"),
            "timm":        _safe_version("timm"),
            "numpy":       _safe_version("numpy"),
            "sklearn":     _safe_version("sklearn"),
            "scipy":       _safe_version("scipy"),
        },
        "cuda":            _cuda_info(),
        "config_hash":     _hash_mapping(config) if config is not None else None,
        "split_manifest_hash": (
            _hash_file(Path(split_manifest_path))
            if split_manifest_path is not None else None
        ),
    }
    return payload


def write_runtime_meta(out_dir: Path | str, **kwargs) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "runtime_meta.json"
    payload = collect_runtime_meta(**kwargs)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path
