"""Optimizer factory.

Default = RectifiedAdam (Table 6 + §4.5 winner). The §4.5 sweep uses:
    RectifiedAdam, AdaBelief, LAMB, LazyAdam, NovoGrad.
"""
from __future__ import annotations

import torch
from torch.optim import Optimizer


def build(name: str, params, lr: float, weight_decay: float) -> Optimizer:
    name = name.lower().replace("-", "").replace("_", "")
    try:
        import torch_optimizer as topt
    except ImportError:
        topt = None
    if name in {"rectifiedadam", "radam"}:
        if topt is not None:
            return topt.RAdam(params, lr=lr, weight_decay=weight_decay)
        return torch.optim.RAdam(params, lr=lr, weight_decay=weight_decay)
    if name == "adabelief":
        assert topt is not None, "pip install torch_optimizer"
        return topt.AdaBelief(params, lr=lr, weight_decay=weight_decay)
    if name == "lamb":
        assert topt is not None, "pip install torch_optimizer"
        return topt.Lamb(params, lr=lr, weight_decay=weight_decay)
    if name == "lazyadam":
        # Not in PyTorch proper; approximate with Adam (same behaviour on
        # dense parameters; LazyAdam only differs on sparse gradients).
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "novograd":
        assert topt is not None, "pip install torch_optimizer"
        return topt.NovoGrad(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")
