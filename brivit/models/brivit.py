"""Brivit main model (paper §3.3, Fig. 4).

Requirements met:
    (#23) Main architecture kept: ViT with symmetric dual-channel attention.
    (#25) Forward supports exporting attention maps / intermediate features
          via `forward(..., return_attn=True)`.
    (#26) Pretrained-weight loading modes are explicit ("partial" vs "full")
          with a structured log describing what copied / what was skipped.
    (#32) `use_dual_channel_attn` switch toggles between the proposed
          dual-channel attention and a plain single-path MHA (ablation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


# ===================================================================== config
@dataclass
class ViTConfig:
    img_size: int = 224
    patch_size: int = 32
    in_channels: int = 3
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    drop_path: float = 0.1
    qkv_bias: bool = True
    num_classes: int = 2
    use_dual_channel_attn: bool = True


# ===================================================================== blocks
class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x.div(keep) * x.new_empty(shape).bernoulli_(keep)


class PatchEmbed(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        assert cfg.img_size % cfg.patch_size == 0
        self.n_patches = (cfg.img_size // cfg.patch_size) ** 2
        self.proj = nn.Conv2d(cfg.in_channels, cfg.embed_dim,
                              kernel_size=cfg.patch_size,
                              stride=cfg.patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)            # (B, N, E)


class MHA(nn.Module):
    """Returns (output, attention_weights). `need_weights` toggles cost."""
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, need_weights: bool = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out, (attn if need_weights else None)


class DualChannelAttn(nn.Module):
    """Two parallel MHA pathways, fused via element-wise sum + LayerNorm (§3.3)."""
    def __init__(self, dim, heads, qkv_bias, drop_path):
        super().__init__()
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.attn_a = MHA(dim, heads, qkv_bias)
        self.attn_b = MHA(dim, heads, qkv_bias)
        self.dp_a = DropPath(drop_path)
        self.dp_b = DropPath(drop_path)
        self.fuse_norm = nn.LayerNorm(dim)

    def forward(self, x, need_weights=False):
        a, Aa = self.attn_a(self.norm_a(x), need_weights)
        b, Ab = self.attn_b(self.norm_b(x), need_weights)
        out = self.fuse_norm(self.dp_a(a) + self.dp_b(b))
        return out, (Aa, Ab) if need_weights else None


class SingleAttn(nn.Module):
    """Baseline single-path attention for ablation."""
    def __init__(self, dim, heads, qkv_bias, drop_path):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = MHA(dim, heads, qkv_bias)
        self.dp = DropPath(drop_path)

    def forward(self, x, need_weights=False):
        out, A = self.attn(self.norm(x), need_weights)
        return self.dp(out), A if need_weights else None


class MLP(nn.Module):
    def __init__(self, dim, hidden, drop_path):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act1 = nn.GELU()
        self.dp1 = DropPath(drop_path)
        self.fc2 = nn.Linear(hidden, dim)
        self.act2 = nn.GELU()
        self.dp2 = DropPath(drop_path)

    def forward(self, x):
        x = self.dp1(self.act1(self.fc1(x)))
        return self.dp2(self.act2(self.fc2(x)))


class EncoderBlock(nn.Module):
    def __init__(self, cfg: ViTConfig, drop_path: float):
        super().__init__()
        Attn = DualChannelAttn if cfg.use_dual_channel_attn else SingleAttn
        self.attn = Attn(cfg.embed_dim, cfg.num_heads, cfg.qkv_bias, drop_path)
        self.norm_m = nn.LayerNorm(cfg.embed_dim)
        self.mlp = MLP(cfg.embed_dim,
                       int(cfg.embed_dim * cfg.mlp_ratio),
                       drop_path)
        self.dp = DropPath(drop_path)

    def forward(self, x, need_weights=False):
        out, A = self.attn(x, need_weights)
        x = x + out
        x = x + self.dp(self.mlp(self.norm_m(x)))
        return x, A


# --------------------- attention normalisation helper ---------------------
def _normalize_attn(A, *, dual: bool) -> dict:
    """Turn whatever an EncoderBlock produced for its attention into a dict
    with well-known keys. Shapes are uniformly (B, heads, N, N).

    For DualChannelAttn:
        A is a tuple (Aa, Ab) -> {"channel_a","channel_b","fused"}.
        "fused" = 0.5 * (Aa + Ab) elementwise — the simplest fusion that
        preserves the attention's probability-distribution shape per row.

    For SingleAttn:
        A is a single tensor -> {"single"}.
    """
    if A is None:
        return {}
    if dual:
        Aa, Ab = A
        return {
            "channel_a": Aa,
            "channel_b": Ab,
            "fused": 0.5 * (Aa + Ab),
        }
    return {"single": A}


# ===================================================================== model
class Brivit(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        N = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, cfg.embed_dim))
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path, cfg.depth)]
        self.blocks = nn.ModuleList(EncoderBlock(cfg, dpr[i]) for i in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        self._init_weights()
        # pretrained-load log (#26)
        self.pretrained_load_log: dict = {"mode": None, "copied": [], "skipped": []}

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------- forward
    def forward(self, x: torch.Tensor,
                return_dict: bool = False,
                return_attention: bool = False,
                return_features: bool = False,
                all_layers: bool = False):
        """Run inference.

        Default (no flags):
            Returns ``logits`` — a Tensor of shape (B, num_classes). This path
            has zero attention-weight overhead, so training speed is unchanged.

        With any of the flags:
            Returns a dict with the keys described below. Passing
            ``return_dict=True`` alone is equivalent to turning on every flag.

        Dict keys (always present when ``return_dict=True``):

            "logits"    — (B, num_classes) Tensor.

            "features"  — (B, embed_dim) Tensor. This is the CLS-token
                           representation at the input of the classification
                           head, i.e. the vector that ``self.head`` consumes.

            "attention" — dict with the last encoder block's attention maps.
                           For the dual-channel model this dict has three
                           entries:
                               "channel_a" : (B, heads, N, N)
                               "channel_b" : (B, heads, N, N)
                               "fused"     : (B, heads, N, N)   elementwise mean
                           For the single-path (ablation) model:
                               "single"    : (B, heads, N, N)
                           Here N = number of tokens = n_patches + 1
                           (patches + CLS), heads = cfg.num_heads.

            "attention_all_layers" — only when ``all_layers=True``. A list of
                                     per-layer dicts in the same schema as
                                     "attention". Length == cfg.depth.

        Args:
            return_dict:        Shortcut; equivalent to turning on every flag.
            return_attention:   Collect attention maps and include them in the
                                returned dict.
            return_features:    Include the pre-head CLS vector.
            all_layers:         Attach per-layer attention as well.
        """
        want_attn = return_dict or return_attention or all_layers
        want_feats = return_dict or return_features

        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        attns_per_layer: list = [] if want_attn else None
        for blk in self.blocks:
            x, A = blk(x, need_weights=want_attn)
            if want_attn:
                attns_per_layer.append(A)
        x = self.norm(x)
        feats = x[:, 0]
        logits = self.head(feats)

        if not (want_attn or want_feats or return_dict):
            return logits

        out: dict = {"logits": logits}
        if want_feats:
            out["features"] = feats
        if want_attn:
            out["attention"] = _normalize_attn(
                attns_per_layer[-1],
                dual=self.cfg.use_dual_channel_attn,
            )
            if all_layers:
                out["attention_all_layers"] = [
                    _normalize_attn(a, dual=self.cfg.use_dual_channel_attn)
                    for a in attns_per_layer
                ]
        return out

    # ------------------------------------------------------- pretrained load
    def load_pretrained(self, state_dict: dict, mode: str = "partial") -> dict:
        """Explicit pretrained loading.

        mode = 'partial' : copy only tensors whose shape matches (default).
        mode = 'full'    : require all tensors to match; raise otherwise.

        Returns the load log, also stored on self.pretrained_load_log (#26).
        """
        own = self.state_dict()
        log = {"mode": mode, "copied": [], "skipped": []}
        if mode == "full":
            missing = [k for k in own if k not in state_dict]
            extra = [k for k in state_dict if k not in own]
            if missing or extra:
                raise RuntimeError(
                    f"full pretrained load failed: missing={missing[:5]}..., "
                    f"extra={extra[:5]}..."
                )
            for k in own:
                if own[k].shape != state_dict[k].shape:
                    raise RuntimeError(
                        f"shape mismatch for {k}: "
                        f"{own[k].shape} vs {state_dict[k].shape}"
                    )
                own[k] = state_dict[k]
                log["copied"].append(k)
        elif mode == "partial":
            for k, v in state_dict.items():
                if k in own and own[k].shape == v.shape:
                    own[k] = v
                    log["copied"].append(k)
                else:
                    log["skipped"].append(k)
        else:
            raise ValueError(f"Unknown pretrained mode: {mode}")
        self.load_state_dict(own)
        self.pretrained_load_log = log
        return log


# ===================================================================== factory
def build_from_config(cfg: dict) -> Brivit:
    mc = cfg["model"]
    use_dca = cfg.get("ablation", {}).get("use_dual_channel_attn",
                                          mc["dual_channel_attention"])
    vit_cfg = ViTConfig(
        img_size=mc["input_size"][0],
        patch_size=mc["patch_size"],
        in_channels=mc["in_channels"],
        embed_dim=mc["embed_dim"],
        depth=mc["depth"],
        num_heads=mc["num_heads"],
        mlp_ratio=mc["mlp_ratio"],
        drop_path=mc["drop_path_rate"],
        qkv_bias=mc["qkv_bias"],
        num_classes=mc["num_classes"],
        use_dual_channel_attn=use_dca,
    )
    model = Brivit(vit_cfg)
    if mc["pretrained"]["enabled"]:
        try:
            import timm
            src = mc["pretrained"]["source"]
            assert src.startswith("timm:"), f"unsupported source {src}"
            backbone = timm.create_model(src.split(":", 1)[1], pretrained=True)
            model.load_pretrained(backbone.state_dict(),
                                  mode=mc["pretrained"]["mode"])
        except Exception as e:
            model.pretrained_load_log = {"mode": "FAILED", "error": str(e)}
    return model
