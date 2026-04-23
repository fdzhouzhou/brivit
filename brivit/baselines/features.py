"""Baseline feature extractors (#31).

Two feature modes, both derived directly from waveforms:

    classical: MFCC mean + std + delta statistics        -> SVM / RandomForest
    cnn      : log-Mel spectrogram (single-channel image) -> 2D CNN / CRNN

All baselines consume the SAME TORGO subject-disjoint split as the main model
(#29), so they cannot overfit to train/test leakage the main model was immune to.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# --------------------------------------------------------------- utilities
def _standardize(path: str, sr: int, clip_len: int) -> torch.Tensor:
    import torchaudio  # lazy
    wav, src_sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if src_sr != sr:
        wav = torchaudio.functional.resample(wav, src_sr, sr)
    wav = wav.squeeze(0)
    if wav.numel() < clip_len:
        wav = torch.nn.functional.pad(wav, (0, clip_len - wav.numel()))
    elif wav.numel() > clip_len:
        s = (wav.numel() - clip_len) // 2
        wav = wav[s: s + clip_len]
    return wav


# --------------------------------------------------------------- classical
@dataclass
class ClassicalFeatureConfig:
    sr: int = 16000
    clip_s: float = 3.0
    n_mfcc: int = 20
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 64


def mfcc_stats(wav: torch.Tensor, cfg: ClassicalFeatureConfig) -> np.ndarray:
    """mean + std + delta-mean + delta-std concatenated across MFCC bins."""
    import torchaudio.transforms as T  # lazy
    mfcc_t = T.MFCC(
        sample_rate=cfg.sr, n_mfcc=cfg.n_mfcc,
        melkwargs={"n_fft": cfg.n_fft, "win_length": cfg.win_length,
                   "hop_length": cfg.hop_length, "n_mels": cfg.n_mels},
    )
    m = mfcc_t(wav.unsqueeze(0)).squeeze(0)                # (n_mfcc, T)
    dm = torch.diff(m, n=1, dim=-1)
    feat = torch.cat([m.mean(-1), m.std(-1),
                      dm.mean(-1), dm.std(-1)], dim=-1)    # (4*n_mfcc,)
    return feat.numpy().astype(np.float32)


def extract_classical(samples: list[dict],
                      cfg: ClassicalFeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    clip_len = int(cfg.sr * cfg.clip_s)
    for s in samples:
        wav = _standardize(s["path"], cfg.sr, clip_len)
        X.append(mfcc_stats(wav, cfg))
        y.append(int(s["label"]))
    return np.stack(X, 0), np.asarray(y, dtype=np.int64)


# --------------------------------------------------------------- CNN features
@dataclass
class LogMelConfig:
    sr: int = 16000
    clip_s: float = 3.0
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 64


def logmel(wav: torch.Tensor, cfg: LogMelConfig) -> torch.Tensor:
    import torchaudio.transforms as T  # lazy
    mel = T.MelSpectrogram(sample_rate=cfg.sr, n_fft=cfg.n_fft,
                           win_length=cfg.win_length,
                           hop_length=cfg.hop_length,
                           n_mels=cfg.n_mels, power=2.0)(wav.unsqueeze(0))
    return torch.log1p(mel).squeeze(0)                     # (n_mels, T)


def build_logmel_tensor(sample: dict, cfg: LogMelConfig) -> torch.Tensor:
    wav = _standardize(sample["path"], cfg.sr, int(cfg.sr * cfg.clip_s))
    return logmel(wav, cfg).unsqueeze(0)                   # (1, F, T)
