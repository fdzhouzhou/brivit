"""Tests for the attention / feature export contract.

Integration test — needs torch. Runs on CPU (tiny model, no CUDA).
"""
from __future__ import annotations

import importlib.util

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        importlib.util.find_spec("torch") is None,
        reason="torch not installed; CPU-only install needs `.[cpu]`.",
    ),
]


def _tiny_cfg(dual: bool):
    from brivit.models.brivit import ViTConfig
    # Small enough to run in milliseconds on CPU.
    return ViTConfig(
        img_size=32, patch_size=16, in_channels=3,
        embed_dim=32, depth=2, num_heads=4,
        mlp_ratio=2.0, drop_path=0.0, qkv_bias=True,
        num_classes=2, use_dual_channel_attn=dual,
    )


def _make_input():
    import torch
    return torch.randn(2, 3, 32, 32)


def test_default_forward_returns_tensor():
    """No flags -> model returns a plain Tensor (the training hot path)."""
    import torch
    from brivit.models.brivit import Brivit
    model = Brivit(_tiny_cfg(dual=True)).eval()
    with torch.no_grad():
        out = model(_make_input())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 2)


def test_return_dict_dual_channel_exposes_all_keys():
    """(revision item 2)
        out = model(x, return_dict=True)
        assert "logits" in out and "features" in out and "attention" in out
    """
    import torch
    from brivit.models.brivit import Brivit
    cfg = _tiny_cfg(dual=True)
    model = Brivit(cfg).eval()
    with torch.no_grad():
        out = model(_make_input(), return_dict=True)
    assert isinstance(out, dict)
    assert "logits" in out
    assert "features" in out
    assert "attention" in out

    # logits shape
    assert out["logits"].shape == (2, cfg.num_classes)
    # features = pre-head CLS vector
    assert out["features"].shape == (2, cfg.embed_dim)

    # attention dict has the dual-channel keys
    attn = out["attention"]
    assert set(attn) == {"channel_a", "channel_b", "fused"}
    n_tokens = (cfg.img_size // cfg.patch_size) ** 2 + 1
    expected = (2, cfg.num_heads, n_tokens, n_tokens)
    for k, m in attn.items():
        assert m.shape == expected, f"{k} has shape {m.shape}, want {expected}"


def test_return_dict_single_channel_exposes_single_attn():
    import torch
    from brivit.models.brivit import Brivit
    cfg = _tiny_cfg(dual=False)
    model = Brivit(cfg).eval()
    with torch.no_grad():
        out = model(_make_input(), return_dict=True)
    attn = out["attention"]
    assert set(attn) == {"single"}
    n_tokens = (cfg.img_size // cfg.patch_size) ** 2 + 1
    assert attn["single"].shape == (2, cfg.num_heads, n_tokens, n_tokens)


def test_all_layers_attention_has_correct_length():
    import torch
    from brivit.models.brivit import Brivit
    cfg = _tiny_cfg(dual=True)
    model = Brivit(cfg).eval()
    with torch.no_grad():
        out = model(_make_input(), return_dict=True, all_layers=True)
    assert "attention_all_layers" in out
    assert len(out["attention_all_layers"]) == cfg.depth
    for layer_attn in out["attention_all_layers"]:
        assert set(layer_attn) == {"channel_a", "channel_b", "fused"}


def test_selective_flags_dont_require_full_return_dict():
    """return_features=True alone should still give back a dict with
    `features` + `logits`, and attention-less behaviour should be cheap."""
    import torch
    from brivit.models.brivit import Brivit
    model = Brivit(_tiny_cfg(dual=True)).eval()
    with torch.no_grad():
        out = model(_make_input(), return_features=True)
    assert isinstance(out, dict)
    assert "features" in out and "logits" in out


def test_dual_and_single_output_shapes_match():
    """dual-channel and single-path models must agree on all public shapes —
    logits, features, and per-layer attention map size (revision #9)."""
    import torch
    from brivit.models.brivit import Brivit
    cfg_dual = _tiny_cfg(dual=True)
    cfg_sng  = _tiny_cfg(dual=False)

    m_dual = Brivit(cfg_dual).eval()
    m_sng  = Brivit(cfg_sng).eval()

    with torch.no_grad():
        x = _make_input()
        od = m_dual(x, return_dict=True)
        os_ = m_sng(x, return_dict=True)

    assert od["logits"].shape == os_["logits"].shape
    assert od["features"].shape == os_["features"].shape

    # Same token count / head count for attention even though the key schema differs.
    any_dual_map = next(iter(od["attention"].values()))
    any_sng_map  = next(iter(os_["attention"].values()))
    assert any_dual_map.shape == any_sng_map.shape
