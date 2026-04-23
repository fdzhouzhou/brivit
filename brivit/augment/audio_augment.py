"""Seven audio augmentation operations from paper Section 3.1.

Waveform-level      : noise / pitch / speed / time-shift
Spectrogram-level   : freq-mask / time-mask / time-warp

Each operation is applied independently with probability `p_each`.
Attached to training dataset only (enforced upstream in brivit.data.dataset).

Controlled by the ablation switch `ablation.use_augment`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ======================== waveform-level ========================
def add_noise(wav: torch.Tensor, snr_db_range=(10.0, 30.0)) -> torch.Tensor:
    snr_db = random.uniform(*snr_db_range)
    power = wav.pow(2).mean().clamp(min=1e-10)
    noise_power = power / (10 ** (snr_db / 10))
    return wav + torch.randn_like(wav) * noise_power.sqrt()


def pitch_shift(wav: torch.Tensor, sr: int, semitone_range=(-2, 2)) -> torch.Tensor:
    import torchaudio  # lazy — see dataset.py rationale
    n_steps = random.uniform(*semitone_range)
    x = wav.unsqueeze(0) if wav.dim() == 1 else wav
    out = torchaudio.functional.pitch_shift(x, sr, n_steps)
    return out.squeeze(0) if wav.dim() == 1 else out


def speed_perturb(wav: torch.Tensor, sr: int, factor_range=(0.9, 1.1)) -> torch.Tensor:
    import torchaudio  # lazy
    factor = random.uniform(*factor_range)
    x = wav.unsqueeze(0) if wav.dim() == 1 else wav
    effects = [["speed", f"{factor:.3f}"], ["rate", str(sr)]]
    out, _ = torchaudio.sox_effects.apply_effects_tensor(x, sr, effects)
    return out.squeeze(0) if wav.dim() == 1 else out


def time_shift(wav: torch.Tensor, max_shift_ratio=0.1) -> torch.Tensor:
    n = wav.shape[-1]
    shift = int(random.uniform(-max_shift_ratio, max_shift_ratio) * n)
    if shift == 0:
        return wav
    out = torch.zeros_like(wav)
    if shift > 0:
        out[..., shift:] = wav[..., :-shift]
    else:
        out[..., :shift] = wav[..., -shift:]
    return out


# ======================== spectrogram-level ========================
def freq_mask(spec: torch.Tensor, max_width: int, num_masks: int) -> torch.Tensor:
    out = spec.clone()
    F_ = out.shape[-2]
    for _ in range(num_masks):
        w = random.randint(0, max_width)
        if 0 < w < F_:
            f0 = random.randint(0, F_ - w)
            out[..., f0:f0 + w, :] = 0.0
    return out


def time_mask(spec: torch.Tensor, max_width: int, num_masks: int) -> torch.Tensor:
    out = spec.clone()
    T_ = out.shape[-1]
    for _ in range(num_masks):
        w = random.randint(0, max_width)
        if 0 < w < T_:
            t0 = random.randint(0, T_ - w)
            out[..., :, t0:t0 + w] = 0.0
    return out


def time_warp(spec: torch.Tensor, max_warp: int) -> torch.Tensor:
    if max_warp <= 0 or spec.shape[-1] < 4:
        return spec
    added_batch = False
    if spec.dim() == 2:
        spec = spec.unsqueeze(0).unsqueeze(0)
        added_batch = True
    elif spec.dim() == 3:
        spec = spec.unsqueeze(0)
        added_batch = True
    B, C, F_, T_ = spec.shape
    anchor = random.randint(max_warp, T_ - max_warp - 1)
    warp = random.randint(-max_warp, max_warp)
    new_idx = torch.empty(T_)
    new_idx[:anchor + 1] = torch.linspace(0, anchor + warp, anchor + 1)
    new_idx[anchor + 1:] = torch.linspace(anchor + warp, T_ - 1, T_ - anchor - 1)
    gx = (new_idx / (T_ - 1)) * 2.0 - 1.0
    gy = torch.linspace(-1.0, 1.0, F_)
    GY, GX = torch.meshgrid(gy, gx, indexing="ij")
    grid = torch.stack([GX, GY], -1).unsqueeze(0).expand(B, -1, -1, -1).to(spec.device)
    out = F.grid_sample(spec, grid, mode="bilinear",
                        padding_mode="border", align_corners=True)
    if added_batch:
        out = out.squeeze(0)
        if out.dim() == 3 and out.shape[0] == 1:
            out = out.squeeze(0)
    return out


# ======================== wrappers ========================
@dataclass
class AugmentConfig:
    sr: int = 16000
    p_each: float = 0.5
    # waveform
    snr_db_range: tuple = (10.0, 30.0)
    semitone_range: tuple = (-2, 2)
    speed_range: tuple = (0.9, 1.1)
    max_shift_ratio: float = 0.1
    use_add_noise: bool = True
    use_pitch: bool = True
    use_speed: bool = True
    use_time_shift: bool = True
    # spectrogram
    freq_mask_width: int = 15
    freq_mask_num: int = 2
    time_mask_width: int = 30
    time_mask_num: int = 2
    time_warp_max: int = 5
    use_freq_mask: bool = True
    use_time_mask: bool = True
    use_time_warp: bool = True


class WaveformAugment:
    def __init__(self, cfg: AugmentConfig):
        self.cfg = cfg

    def _maybe(self) -> bool:
        return random.random() < self.cfg.p_each

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        if c.use_add_noise and self._maybe():
            wav = add_noise(wav, c.snr_db_range)
        if c.use_pitch and self._maybe():
            wav = pitch_shift(wav, c.sr, c.semitone_range)
        if c.use_speed and self._maybe():
            wav = speed_perturb(wav, c.sr, c.speed_range)
        if c.use_time_shift and self._maybe():
            wav = time_shift(wav, c.max_shift_ratio)
        return wav


class SpectrogramAugment:
    def __init__(self, cfg: AugmentConfig):
        self.cfg = cfg

    def _maybe(self) -> bool:
        return random.random() < self.cfg.p_each

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        if c.use_freq_mask and self._maybe():
            spec = freq_mask(spec, c.freq_mask_width, c.freq_mask_num)
        if c.use_time_mask and self._maybe():
            spec = time_mask(spec, c.time_mask_width, c.time_mask_num)
        if c.use_time_warp and self._maybe():
            spec = time_warp(spec, c.time_warp_max)
        return spec


def build_from_config(cfg: dict) -> tuple[WaveformAugment | None,
                                          SpectrogramAugment | None]:
    """Honour the ablation switch `ablation.use_augment`.  Returns (None, None)
    if augmentation is disabled — the dataset then never perturbs inputs."""
    if not cfg.get("ablation", {}).get("use_augment", True) \
            or not cfg["augmentation"]["enabled"]:
        return None, None
    ac = AugmentConfig(
        sr=cfg["audio"]["sampling_rate"],
        p_each=cfg["augmentation"]["probability_each"],
        snr_db_range=tuple(cfg["augmentation"]["waveform"]["add_noise"]["snr_db_range"]),
        semitone_range=tuple(cfg["augmentation"]["waveform"]["pitch_shift"]["semitone_range"]),
        speed_range=tuple(cfg["augmentation"]["waveform"]["speed_perturb"]["factor_range"]),
        max_shift_ratio=cfg["augmentation"]["waveform"]["time_shift"]["max_shift_ratio"],
        use_add_noise=cfg["augmentation"]["waveform"]["add_noise"]["enabled"],
        use_pitch=cfg["augmentation"]["waveform"]["pitch_shift"]["enabled"],
        use_speed=cfg["augmentation"]["waveform"]["speed_perturb"]["enabled"],
        use_time_shift=cfg["augmentation"]["waveform"]["time_shift"]["enabled"],
        freq_mask_width=cfg["augmentation"]["spectrogram"]["freq_mask"]["max_width"],
        freq_mask_num=cfg["augmentation"]["spectrogram"]["freq_mask"]["num_masks"],
        time_mask_width=cfg["augmentation"]["spectrogram"]["time_mask"]["max_width"],
        time_mask_num=cfg["augmentation"]["spectrogram"]["time_mask"]["num_masks"],
        time_warp_max=cfg["augmentation"]["spectrogram"]["time_warp"]["max_warp"],
        use_freq_mask=cfg["augmentation"]["spectrogram"]["freq_mask"]["enabled"],
        use_time_mask=cfg["augmentation"]["spectrogram"]["time_mask"]["enabled"],
        use_time_warp=cfg["augmentation"]["spectrogram"]["time_warp"]["enabled"],
    )
    return WaveformAugment(ac), SpectrogramAugment(ac)
