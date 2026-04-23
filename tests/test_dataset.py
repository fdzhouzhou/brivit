"""Tests for the train-only-augmentation contract (#17, #18, #46).

Marked `integration` because it imports `brivit.data.dataset`, which needs
torch (for `torch.utils.data.Dataset` as a base class). If torch is NOT
installed in the current environment this whole module is skipped cleanly
rather than failing — the CPU-only test suite (`pytest -m "not gpu"`) will
still include it when torch is available, and skip it when not.

This test does NOT call torchaudio, so it works on pure CPU installs.
"""
from __future__ import annotations

import importlib.util

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        importlib.util.find_spec("torch") is None,
        reason="torch not installed in this env; CPU-only install needs `.[cpu]`.",
    ),
]


@pytest.fixture(scope="module")
def TorgoDataset():
    # Deferred import so collection works even without torch.
    from brivit.data.dataset import TorgoDataset
    return TorgoDataset


def test_train_accepts_augmentation(TorgoDataset):
    dummy_prep = lambda wav: wav
    aug = lambda w: w
    # Must not raise.
    TorgoDataset([], dummy_prep, split="train",
                 waveform_aug=aug, spec_aug=aug)


def test_val_rejects_waveform_aug(TorgoDataset):
    dummy_prep = lambda wav: wav
    with pytest.raises(AssertionError):
        TorgoDataset([], dummy_prep, split="val",
                     waveform_aug=lambda w: w)


def test_val_rejects_spec_aug(TorgoDataset):
    dummy_prep = lambda wav: wav
    with pytest.raises(AssertionError):
        TorgoDataset([], dummy_prep, split="val",
                     spec_aug=lambda w: w)


def test_test_rejects_waveform_aug(TorgoDataset):
    dummy_prep = lambda wav: wav
    with pytest.raises(AssertionError):
        TorgoDataset([], dummy_prep, split="test",
                     waveform_aug=lambda w: w)


def test_test_rejects_spec_aug(TorgoDataset):
    dummy_prep = lambda wav: wav
    with pytest.raises(AssertionError):
        TorgoDataset([], dummy_prep, split="test",
                     spec_aug=lambda w: w)


def test_bad_split_name(TorgoDataset):
    dummy_prep = lambda wav: wav
    with pytest.raises(ValueError):
        TorgoDataset([], dummy_prep, split="holdout")
