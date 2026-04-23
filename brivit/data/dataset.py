"""TORGO PyTorch Dataset.

Key contract (from revision):
    (#17) Train allows augmentation; val/test MUST NOT be augmented.
    (#18) Augmented samples never participate in metric aggregation.
    (#19) Unified sampling rate / channels / clip duration for every split.
    (#46) Evaluation always runs on original (non-augmented) test samples.

This module is split-protocol-agnostic: it consumes a list of sample dicts
(from `brivit.data.splits`) and returns tensors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset


def _load_and_standardize(path: str, target_sr: int, clip_len: int) -> torch.Tensor:
    """Load -> mono -> resample -> pad/center-crop to `clip_len` samples.

    torchaudio is imported lazily so that module-level ``import`` of this file
    does not trigger loading torchaudio's native libraries — this keeps the
    unit tests in tests/test_dataset.py runnable on CPU-only installs where
    torchaudio is not present or its CUDA counterparts cannot be resolved.
    """
    import torchaudio  # lazy
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.squeeze(0)
    if wav.numel() < clip_len:
        wav = torch.nn.functional.pad(wav, (0, clip_len - wav.numel()))
    elif wav.numel() > clip_len:
        start = (wav.numel() - clip_len) // 2
        wav = wav[start:start + clip_len]
    return wav


class TorgoDataset(Dataset):
    """Serves (feature_map, label, meta).

    Args:
        samples: sample-dict list from `discover_torgo` / `samples_for_fold`.
        preprocessor: callable(waveform 1-D tensor) -> 3-D (C,H,W) float.
        split: 'train' | 'val' | 'test'. Drives whether augmentation runs.
        sample_rate, clip_s: audio standardization.
        waveform_aug, spec_aug: callables, applied ONLY when split == 'train'.
        return_meta: if True, also returns sample metadata (path, speaker, ...).
    """
    def __init__(self,
                 samples: Sequence[dict],
                 preprocessor: Callable[[torch.Tensor], torch.Tensor],
                 split: str,
                 sample_rate: int = 16000,
                 clip_s: float = 3.0,
                 waveform_aug: Callable | None = None,
                 spec_aug: Callable | None = None,
                 return_meta: bool = False):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        if split != "train" and (waveform_aug is not None or spec_aug is not None):
            # Hard enforcement of (#17, #18, #46)
            raise AssertionError(
                "Augmentation must NOT be attached to val/test datasets."
            )
        self.samples = list(samples)
        self.prep = preprocessor
        self.split = split
        self.sample_rate = sample_rate
        self.clip_len = int(sample_rate * clip_s)
        self.waveform_aug = waveform_aug
        self.spec_aug = spec_aug
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        wav = _load_and_standardize(s["path"], self.sample_rate, self.clip_len)
        if self.split == "train" and self.waveform_aug is not None:
            wav = self.waveform_aug(wav)
            # re-normalize length in case speed-perturb changed it
            if wav.numel() < self.clip_len:
                wav = torch.nn.functional.pad(wav, (0, self.clip_len - wav.numel()))
            elif wav.numel() > self.clip_len:
                wav = wav[:self.clip_len]
        x = self.prep(wav)                    # (C,H,W)
        if self.split == "train" and self.spec_aug is not None:
            x = self.spec_aug(x)
        if self.return_meta:
            return x.float(), int(s["label"]), s
        return x.float(), int(s["label"])


def make_loaders(samples_by_split: dict[str, list[dict]],
                 preprocessor,
                 batch_size: int,
                 num_workers: int,
                 sample_rate: int,
                 clip_s: float,
                 waveform_aug=None,
                 spec_aug=None) -> dict[str, DataLoader]:
    """Build {train, val, test} DataLoaders with the correct augmentation policy."""
    loaders = {}
    for split, samples in samples_by_split.items():
        ds = TorgoDataset(
            samples, preprocessor, split=split,
            sample_rate=sample_rate, clip_s=clip_s,
            waveform_aug=waveform_aug if split == "train" else None,
            spec_aug=spec_aug if split == "train" else None,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=(split == "train"),
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
