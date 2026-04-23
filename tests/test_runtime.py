"""Runtime meta snapshot tests (revision #10).

Pure-Python — marked `unit`.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from brivit.utils.runtime_meta import (collect_runtime_meta,
                                       write_runtime_meta)

pytestmark = pytest.mark.unit


def test_collect_has_required_keys():
    payload = collect_runtime_meta(config={"a": 1}, component="test")
    for k in ("timestamp", "python", "platform", "os", "package_version",
              "git_commit", "versions", "cuda", "config_hash",
              "component", "hostname"):
        assert k in payload, f"missing key: {k}"
    # config hash deterministic
    h1 = collect_runtime_meta(config={"a": 1}, component="t")["config_hash"]
    h2 = collect_runtime_meta(config={"a": 1}, component="t")["config_hash"]
    assert h1 == h2
    h3 = collect_runtime_meta(config={"a": 2}, component="t")["config_hash"]
    assert h1 != h3


def test_write_produces_json_file(tmp_path):
    p = write_runtime_meta(tmp_path, config={"x": "y"}, component="unit_test")
    assert p.is_file()
    payload = json.loads(p.read_text())
    assert payload["component"] == "unit_test"
    # versions dict must exist even when some libs absent
    assert "torch" in payload["versions"]
    assert "torchaudio" in payload["versions"]


def test_cuda_info_structure():
    payload = collect_runtime_meta()
    cuda = payload["cuda"]
    assert set(cuda) >= {"available", "device_count", "current_device_name"}
    assert isinstance(cuda["available"], bool)
    assert isinstance(cuda["device_count"], int)
