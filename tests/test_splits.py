"""Smoke tests for the speaker-disjoint split logic.

These tests do NOT require the actual TORGO dataset; they build a synthetic
speaker index and verify the algorithm's invariants.

Pure-Python — marked `unit`, runs in the CPU-only test set.
"""
from __future__ import annotations

import pytest

from brivit.data.splits import (Fold, leave_one_speaker_out,
                                speaker_stratified_kfold)

pytestmark = pytest.mark.unit


def _synthetic_speakers():
    """12 speakers balanced over (gender x label)."""
    spk = {}
    for g in ("F", "M"):
        for label, prefix in [(1, ""), (0, "C")]:
            for i in range(3):
                sid = f"{g}{prefix}{i + 1:02d}"
                spk[sid] = {"speaker": sid, "gender": g,
                            "label": label, "n_samples": 10}
    return spk


def test_kfold_disjoint():
    speakers = _synthetic_speakers()
    folds = speaker_stratified_kfold(
        speakers, k=5, val_fraction=0.2,
        stratify_on=["label", "gender"], seed=42,
    )
    assert len(folds) == 5
    all_spk = set(speakers)
    for fold in folds:
        fold.assert_disjoint()     # (#13)
        # every speaker is assigned to exactly one of {train, val, test}
        covered = set(fold.train_speakers) | set(fold.val_speakers) | set(fold.test_speakers)
        assert covered == all_spk, f"fold {fold.index}: missing={all_spk - covered}"


def test_kfold_test_partitions_cover_all_speakers_exactly_once():
    speakers = _synthetic_speakers()
    folds = speaker_stratified_kfold(
        speakers, k=5, val_fraction=0.2,
        stratify_on=["label", "gender"], seed=7,
    )
    # Across folds, each speaker appears in exactly one test set.
    from collections import Counter
    seen = Counter()
    for fold in folds:
        for s in fold.test_speakers:
            seen[s] += 1
    assert all(c == 1 for c in seen.values())
    assert set(seen) == set(speakers)


def test_leave_one_out_produces_n_folds():
    speakers = _synthetic_speakers()
    folds = leave_one_speaker_out(speakers, val_fraction=0.2, seed=0)
    assert len(folds) == len(speakers)
    for f in folds:
        assert len(f.test_speakers) == 1
        f.assert_disjoint()


def test_disjoint_assertion_catches_leakage():
    bad = Fold(index=0, train_speakers=["F01"], val_speakers=[], test_speakers=["F01"])
    with pytest.raises(AssertionError):
        bad.assert_disjoint()


# --------------------------------------------------------------- extended
def test_loso_holdout_speaker_is_unique_per_fold():
    speakers = _synthetic_speakers()
    folds = leave_one_speaker_out(speakers, val_fraction=0.2, seed=0)
    # Every speaker shows up as THE held-out test speaker in exactly one fold.
    seen = {}
    for f in folds:
        assert len(f.test_speakers) == 1
        spk = f.test_speakers[0]
        assert spk not in seen, f"{spk} appears in folds {seen[spk]} and {f.index}"
        seen[spk] = f.index
    assert set(seen.keys()) == set(speakers)


def test_fold_quality_reports_all_required_counts():
    """Revision #6: quality report must expose speakers / samples /
    label / gender counts for each split of each fold."""
    from brivit.data.splits import fold_quality

    speakers = _synthetic_speakers()
    folds = speaker_stratified_kfold(
        speakers, k=5, val_fraction=0.2,
        stratify_on=["label", "gender"], seed=42,
    )
    # Build one synthetic sample per speaker
    samples = [{"path": f"/fake/{s}.wav", "speaker": s,
                "gender": speakers[s]["gender"],
                "label":  speakers[s]["label"]} for s in speakers]
    rec = fold_quality(samples, folds[0])
    for split in ("train", "val", "test"):
        for field in ("speakers", "samples", "pos", "neg", "female", "male"):
            assert f"{split}_{field}" in rec, f"missing {split}_{field}"


def test_empty_split_detected():
    """Hard assertion: empty fold split raises FoldQualityError."""
    from brivit.data.splits import (FoldQualityError, fold_quality,
                                    validate_fold_quality)
    # Make a fold with an empty val split.
    speakers = _synthetic_speakers()
    samples = [{"path": f"/f/{s}.wav", "speaker": s,
                "gender": speakers[s]["gender"],
                "label":  speakers[s]["label"]} for s in speakers]
    bad_fold = Fold(index=0,
                    train_speakers=sorted(speakers)[:-1],
                    val_speakers=[],
                    test_speakers=[sorted(speakers)[-1]])
    rec = fold_quality(samples, bad_fold)
    with pytest.raises(FoldQualityError):
        validate_fold_quality(rec, strict=True)
    # Soft mode just warns (returns issues list)
    issues = validate_fold_quality(rec, strict=False)
    assert any("val" in i for i in issues)


def test_single_class_test_fold_detected():
    """Hard assertion: test with only one class -> FoldQualityError."""
    from brivit.data.splits import (FoldQualityError, fold_quality,
                                    validate_fold_quality)
    speakers = _synthetic_speakers()
    samples = [{"path": f"/f/{s}.wav", "speaker": s,
                "gender": speakers[s]["gender"],
                "label":  speakers[s]["label"]} for s in speakers]
    # Fold whose test speakers all share label=1
    only_pos = [s for s in speakers if speakers[s]["label"] == 1]
    others = [s for s in speakers if s not in only_pos]
    bad_fold = Fold(index=1,
                    train_speakers=others[:-1],
                    val_speakers=[others[-1]],
                    test_speakers=only_pos)
    rec = fold_quality(samples, bad_fold)
    with pytest.raises(FoldQualityError):
        validate_fold_quality(rec, strict=True)
