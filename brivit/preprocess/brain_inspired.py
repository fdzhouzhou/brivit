"""Brain-inspired audio preprocessing (paper §3.2).

Pipeline:
    1. Time-domain features     (envelope / short-term energy, implicit via Mel)
    2. Frequency-domain features (Mel, MFCC, PCEN smoothing)  — spectral branch
    3. LIF spike encoding on normalized time-frequency channels — spike branch
    4. Spike-domain post-processing (neighborhood filter + density mask)
    5. Spike statistics: firing rate, FSL, ISI mean/std, burstiness
    6. Multimodal fusion (concat spectral ⊕ spike stats) + linear re-projection
    7. Resize to (C_out, H, W) ready for patch embedding

Ablation switches (handled in build_from_config):
    use_spectral: keep the Mel/MFCC branch
    use_spike   : run LIF + post-processing + stats
    use_fusion  : combine both branches (if both are on)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpikeConfig:
    sr: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 64
    pcen_smoothing: float = 0.98
    # LIF
    tau_m_ms: float = 20.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    tau_ref_ms: float = 5.0
    dt_ms: float = 1.0
    membrane_decay: float = 0.95
    event_threshold: float = 0.2
    # spike post-processing
    neighborhood_ms: float = 5.0
    min_neighbors: int = 2
    density_percentile: float = 75.0
    stats_window_ms: float = 20.0
    # output
    out_time: int = 224
    out_feat: int = 224
    out_channels: int = 3
    # switches
    use_spectral: bool = True
    use_spike: bool = True
    use_fusion: bool = True


# ----------------------------- spectral branch -----------------------------
class _MelMFCC(nn.Module):
    def __init__(self, cfg: SpikeConfig):
        super().__init__()
        import torchaudio.transforms as T  # lazy
        self.mel = T.MelSpectrogram(sample_rate=cfg.sr, n_fft=cfg.n_fft,
                                    win_length=cfg.win_length,
                                    hop_length=cfg.hop_length,
                                    n_mels=cfg.n_mels, power=2.0)
        self.mfcc = T.MFCC(sample_rate=cfg.sr, n_mfcc=20,
                           melkwargs={"n_fft": cfg.n_fft,
                                      "win_length": cfg.win_length,
                                      "hop_length": cfg.hop_length,
                                      "n_mels": cfg.n_mels})
        self.s = cfg.pcen_smoothing

    def forward(self, wav: torch.Tensor):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        mel = self.mel(wav)
        mfcc = self.mfcc(wav)
        # PCEN-style smoothing
        sm = torch.zeros_like(mel)
        prev = torch.zeros_like(mel[..., 0])
        for t in range(mel.shape[-1]):
            prev = self.s * prev + (1 - self.s) * mel[..., t]
            sm[..., t] = prev
        mel_norm = torch.log1p(mel / (sm + 1e-6))
        return mel_norm.squeeze(0), mfcc.squeeze(0)


# ----------------------------- spike branch -----------------------------
@torch.no_grad()
def _lif_encode(x: torch.Tensor, cfg: SpikeConfig) -> torch.Tensor:
    C, T_ = x.shape
    decay = cfg.membrane_decay
    v = torch.full((C,), cfg.v_reset, device=x.device, dtype=x.dtype)
    ref = torch.zeros(C, device=x.device, dtype=torch.long)
    ref_steps = int(round(cfg.tau_ref_ms / cfg.dt_ms))
    spikes = torch.zeros_like(x)
    for t in range(T_):
        active = ref == 0
        v[active] = decay * v[active] + x[active, t]
        fired = active & (v >= cfg.v_threshold)
        spikes[fired, t] = 1.0
        v[fired] = cfg.v_reset
        ref[fired] = ref_steps
        still_ref = (~active) & (ref > 0)
        ref[still_ref] -= 1
    return spikes


@torch.no_grad()
def _neighborhood_filter(spikes: torch.Tensor, cfg: SpikeConfig) -> torch.Tensor:
    w = int(round(cfg.neighborhood_ms / cfg.dt_ms))
    kernel = torch.ones(1, 1, 2 * w + 1, device=spikes.device)
    count = F.conv1d(spikes.unsqueeze(1), kernel,
                     padding=w).squeeze(1) - spikes
    return spikes * (count >= cfg.min_neighbors).float()


@torch.no_grad()
def _density_mask(spikes: torch.Tensor, cfg: SpikeConfig) -> torch.Tensor:
    """Salient-event detection (paper §3.2).

    A spike is kept only if its local firing rate, computed over a
    ``stats_window_ms`` window, satisfies BOTH:

        * rate >= channel-wise ``density_percentile``   (relative threshold)
        * rate >= ``event_threshold``                    (absolute threshold)

    Both thresholds come from the config. Enforcing the absolute threshold
    means that a channel with low absolute activity cannot generate spurious
    "events" just because SOMETHING in it happens to sit above its 75th
    percentile — the config parameter ``event_threshold`` thereby actually
    influences the output feature (revision #3).
    """
    w = max(1, int(round(cfg.stats_window_ms / cfg.dt_ms)))
    kernel = torch.ones(1, 1, w, device=spikes.device) / w
    rate = F.conv1d(spikes.unsqueeze(1), kernel, padding=w // 2).squeeze(1)
    rel_thr = torch.quantile(rate, cfg.density_percentile / 100.0,
                             dim=-1, keepdim=True)
    abs_thr = float(cfg.event_threshold)
    keep = (rate >= rel_thr) & (rate >= abs_thr)
    return spikes * keep.float()


@torch.no_grad()
def _spike_statistics(spikes: torch.Tensor, cfg: SpikeConfig) -> torch.Tensor:
    C, T_ = spikes.shape
    w = max(1, int(round(cfg.stats_window_ms / cfg.dt_ms)))
    nw = T_ // w
    if nw == 0:
        return torch.zeros(C, 1, 5, device=spikes.device)
    s = spikes[:, : nw * w].view(C, nw, w)
    fr = s.mean(-1)
    first = (s > 0).float().cumsum(-1) == 1
    fsl_idx = first.float().argmax(-1).float() / w
    fsl = torch.where(s.sum(-1) > 0, fsl_idx, torch.ones_like(fsl_idx))
    isi_mean = torch.zeros_like(fr)
    isi_std = torch.zeros_like(fr)
    burst = torch.zeros_like(fr)
    for c in range(C):
        for wk in range(nw):
            t = torch.nonzero(s[c, wk]).flatten().float()
            if t.numel() >= 2:
                isi = t[1:] - t[:-1]
                isi_mean[c, wk] = isi.mean()
                isi_std[c, wk] = isi.std(unbiased=False)
                d = (isi_std[c, wk] + isi_mean[c, wk]).clamp(min=1e-6)
                burst[c, wk] = (isi_std[c, wk] - isi_mean[c, wk]) / d
    return torch.stack([fr, fsl, isi_mean, isi_std, burst], -1)  # (C,W,5)


# ----------------------------- top-level module -----------------------------
class BrainInspiredPreprocessor(nn.Module):
    def __init__(self, cfg: SpikeConfig):
        super().__init__()
        self.cfg = cfg
        # Spectral main front-end (Mel + MFCC) — built only when the ablation
        # switch demands it.
        if cfg.use_spectral:
            self.melmfcc = _MelMFCC(cfg)
        else:
            self.melmfcc = None
        # Spike-only fallback front-end (plain Mel). If the spike branch will
        # run but the spectral branch is disabled, we still need a Mel
        # representation to drive the LIF neurons. Build it ONCE here rather
        # than instantiating a new MelSpectrogram per sample in forward()
        # (revision #7).
        if cfg.use_spike and not cfg.use_spectral:
            import torchaudio.transforms as T  # lazy
            self.mel_fallback = T.MelSpectrogram(
                sample_rate=cfg.sr, n_fft=cfg.n_fft,
                win_length=cfg.win_length, hop_length=cfg.hop_length,
                n_mels=cfg.n_mels, power=2.0,
            )
        else:
            self.mel_fallback = None

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # --------- spectral branch ---------
        spec = None
        if cfg.use_spectral:
            mel, mfcc = self.melmfcc(wav)
            spec = torch.cat([mel, mfcc], dim=0)             # (C_spec, T)

        # --------- spike branch ---------
        spike_stats = None
        if cfg.use_spike:
            # Drive the LIF input from whichever Mel representation is already
            # available — never construct a new module here (revision #7).
            if spec is None:
                # spike-only path: uses the cached fallback.
                mel_only = self.mel_fallback(wav)
                base = mel_only.squeeze(0)
            else:
                base = spec
            norm = (base - base.min()) / (base.max() - base.min() + 1e-8)
            T_ms = int(wav.shape[-1] / cfg.sr * 1000 / cfg.dt_ms)
            up = F.interpolate(norm.unsqueeze(0), size=T_ms,
                               mode="linear", align_corners=False).squeeze(0)
            sp = _lif_encode(up, cfg)
            sp = _neighborhood_filter(sp, cfg)
            sp = _density_mask(sp, cfg)
            spike_stats = _spike_statistics(sp, cfg)          # (C,W,5)

        # --------- fusion & output ---------
        if cfg.use_fusion and (spec is not None) and (spike_stats is not None):
            T_frames = spec.shape[-1]
            aligned = F.interpolate(
                spike_stats.permute(2, 0, 1).unsqueeze(0),
                size=(spec.shape[0], T_frames),
                mode="bilinear", align_corners=False,
            ).squeeze(0).permute(1, 2, 0)                     # (C, T, 5)
            fused = torch.cat([spec.unsqueeze(-1), aligned], dim=-1)  # (C,T,6)
            fused = fused.permute(2, 0, 1)                    # (6, C, T)
        elif spec is not None:
            fused = spec.unsqueeze(0)                         # (1,C,T)
        elif spike_stats is not None:
            fused = spike_stats.permute(2, 0, 1)              # (5,C,W)
        else:
            raise RuntimeError(
                "Both spectral and spike branches disabled — nothing to process."
            )

        fused = F.interpolate(fused.unsqueeze(0),
                              size=(cfg.out_feat, cfg.out_time),
                              mode="bilinear",
                              align_corners=False).squeeze(0)
        # linear projection by averaging groups to hit out_channels
        if fused.shape[0] != cfg.out_channels:
            if fused.shape[0] < cfg.out_channels:
                fused = fused.repeat(
                    (cfg.out_channels // fused.shape[0] + 1, 1, 1)
                )[: cfg.out_channels]
            else:
                groups = fused.shape[0] // cfg.out_channels
                fused = fused[: groups * cfg.out_channels].reshape(
                    cfg.out_channels, groups, cfg.out_feat, cfg.out_time
                ).mean(1)
        return fused


# ----------------------------- factory -----------------------------
def build_from_config(cfg: dict) -> BrainInspiredPreprocessor:
    ab = cfg.get("ablation", {})
    sc = SpikeConfig(
        sr=cfg["audio"]["sampling_rate"],
        n_fft=cfg["audio"]["n_fft"],
        hop_length=cfg["audio"]["hop_length"],
        win_length=cfg["audio"]["win_length"],
        n_mels=cfg["audio"]["n_mels"],
        pcen_smoothing=cfg["audio"]["pcen_smoothing"],
        tau_m_ms=cfg["spike_encoding"]["tau_m_ms"],
        v_threshold=cfg["spike_encoding"]["v_threshold"],
        v_reset=cfg["spike_encoding"]["v_reset"],
        tau_ref_ms=cfg["spike_encoding"]["tau_ref_ms"],
        dt_ms=cfg["spike_encoding"]["dt_ms"],
        membrane_decay=cfg["spike_encoding"]["membrane_decay"],
        event_threshold=cfg["spike_encoding"]["event_threshold"],
        neighborhood_ms=cfg["spike_encoding"]["neighborhood_ms"],
        min_neighbors=cfg["spike_encoding"]["min_neighbors"],
        density_percentile=cfg["spike_encoding"]["density_percentile"],
        stats_window_ms=cfg["spike_encoding"]["stats_window_ms"],
        out_time=cfg["model"]["input_size"][1],
        out_feat=cfg["model"]["input_size"][0],
        out_channels=cfg["feature_fusion"]["out_channels"],
        use_spectral=ab.get("use_spectral", True),
        use_spike=ab.get("use_spike", True),
        use_fusion=ab.get("use_fusion", True),
    )
    return BrainInspiredPreprocessor(sc)
