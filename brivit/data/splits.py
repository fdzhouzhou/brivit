"""Subject-disjoint splitting for TORGO.

Requirements from the revision:
    (#9)  Split by speaker, not by file.
    (#10) Same speaker must live in exactly one of {train, val, test}.
    (#11) Default protocol = speaker-stratified 5-fold CV.
          (Leave-one-speaker-out is also offered.)
    (#12) Stratify on (label, gender) so positive/negative and M/F are balanced.
    (#13) Hard assert: pairwise empty speaker intersections in every fold.
    (#14) Split is fixed and exported as a manifest (JSON). All downstream
          experiments — main model, baselines, ablations — consume this file.
    (#15) Sample-level legacy mode is kept behind `sample_level_legacy` but is
          NOT the default.

TORGO speaker folder convention:
    F01, F03, F04, FC01, FC02, FC03, M01, M02, M03, M04, M05,
    MC01, MC02, MC03, MC04   (names vary slightly by mirror)

The "C" infix ⇒ control speaker (label 0). No "C" ⇒ dysarthric (label 1).
Gender ⇒ first character (F or M).

Implementation is robust to the Kaggle mirror layout:
    TORGO/
       F_Con/ FCxx/ SessionY/ wav_[array|head]Mic/ *.wav
       F_Dys/ Fxx/  ...
       M_Con/ MCxx/ ...
       M_Dys/ Mxx/  ...
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------- speaker id
_SPK_RE = re.compile(r"([FM]C?\d+)", re.IGNORECASE)


def speaker_id_from_path(path: str | Path) -> str:
    """Extract the speaker id from a TORGO file path.

    Returns e.g. 'F01', 'FC02', 'M03'. Raises if no id can be found.
    """
    p = Path(path)
    # Walk parents looking for the TORGO speaker pattern
    for part in p.parts:
        m = _SPK_RE.fullmatch(part)
        if m:
            return m.group(1).upper()
    # Fall back to regex over the string
    m = _SPK_RE.search(str(p))
    if m:
        return m.group(1).upper()
    raise ValueError(f"Cannot extract speaker id from path: {path}")


def label_from_speaker(spk: str) -> int:
    """1 = dysarthric, 0 = control.  Dysarthric speakers have NO 'C' infix."""
    return 0 if "C" in spk.upper() else 1


def gender_from_speaker(spk: str) -> str:
    return "F" if spk.upper().startswith("F") else "M"


# ----------------------------------------------------------- file discovery
def discover_torgo(root: str | Path) -> list[dict]:
    """Enumerate every .wav under a TORGO root, annotated with metadata.

    Returns a list of dicts: {path, speaker, gender, label, class_dir}.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"TORGO root not found: {root}")
    out = []
    for wav in root.rglob("*.wav"):
        try:
            spk = speaker_id_from_path(wav)
        except ValueError:
            continue
        # Class dir is the top-level folder under root (e.g. F_Con)
        rel = wav.relative_to(root)
        class_dir = rel.parts[0] if len(rel.parts) > 1 else ""
        out.append({
            "path": str(wav),
            "speaker": spk,
            "gender": gender_from_speaker(spk),
            "label": label_from_speaker(spk),
            "class_dir": class_dir,
        })
    if not out:
        raise RuntimeError(
            f"No .wav files found under {root}. "
            "Expected TORGO structure with F_Con/F_Dys/M_Con/M_Dys subfolders."
        )
    return out


def speakers_index(samples: Iterable[dict]) -> dict[str, dict]:
    """Collapse the per-sample list into per-speaker records."""
    idx = {}
    for s in samples:
        spk = s["speaker"]
        if spk not in idx:
            idx[spk] = {
                "speaker": spk,
                "gender": s["gender"],
                "label": s["label"],
                "n_samples": 0,
            }
        idx[spk]["n_samples"] += 1
    return idx


# -------------------------------------------------------------------- splits
@dataclass
class Fold:
    index: int
    train_speakers: list[str]
    val_speakers: list[str]
    test_speakers: list[str]

    def assert_disjoint(self) -> None:
        s_tr = set(self.train_speakers)
        s_va = set(self.val_speakers)
        s_te = set(self.test_speakers)
        # (#13) pairwise-disjoint assertions
        if s_tr & s_va:
            raise AssertionError(f"Fold {self.index}: train∩val = {s_tr & s_va}")
        if s_tr & s_te:
            raise AssertionError(f"Fold {self.index}: train∩test = {s_tr & s_te}")
        if s_va & s_te:
            raise AssertionError(f"Fold {self.index}: val∩test = {s_va & s_te}")


@dataclass
class SplitManifest:
    protocol: str
    k_folds: int
    seed: int
    dataset_root: str
    stratify_on: list[str] = field(default_factory=list)
    folds: list[Fold] = field(default_factory=list)
    speakers: dict[str, dict] = field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        with open(p, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "SplitManifest":
        with open(path) as f:
            p = json.load(f)
        folds = [Fold(**fo) for fo in p.pop("folds", [])]
        return cls(folds=folds, **p)


# ----------------------------------------------------- stratified k-fold logic
def _stratum_key(s: dict, keys: Iterable[str]) -> tuple:
    return tuple(s[k] for k in keys)


def speaker_stratified_kfold(speakers: dict[str, dict],
                             k: int,
                             val_fraction: float,
                             stratify_on: list[str],
                             seed: int) -> list[Fold]:
    """Round-robin, stratum-wise assignment of speakers to k folds.

    For each stratum (e.g. F/dys, F/con, M/dys, M/con) speakers are shuffled
    with `seed` and dealt round-robin into k buckets. Then fold i uses
    bucket i as test; val_fraction of the remaining speakers is carved out
    for val (stratum-wise, same shuffle); everyone else trains.
    """
    rng = random.Random(seed)
    strata = defaultdict(list)
    for spk, meta in speakers.items():
        strata[_stratum_key(meta, stratify_on)].append(spk)
    for k_s in strata:
        strata[k_s].sort()
        rng.shuffle(strata[k_s])

    # Distribute each stratum's speakers into k buckets.
    buckets: list[list[str]] = [[] for _ in range(k)]
    for stratum_members in strata.values():
        for i, spk in enumerate(stratum_members):
            buckets[i % k].append(spk)

    folds: list[Fold] = []
    all_speakers = set(speakers)
    for i in range(k):
        test = sorted(buckets[i])
        remaining = sorted(all_speakers - set(test))
        # Carve val from remaining, stratum-aware.
        remaining_strata = defaultdict(list)
        for spk in remaining:
            remaining_strata[_stratum_key(speakers[spk], stratify_on)].append(spk)
        val = []
        rng2 = random.Random(seed + 1000 + i)   # deterministic per fold
        for members in remaining_strata.values():
            members = sorted(members)
            rng2.shuffle(members)
            n_val = max(1, int(round(len(members) * val_fraction)))
            val.extend(members[:n_val])
        val = sorted(val)
        train = sorted(set(remaining) - set(val))

        f = Fold(index=i, train_speakers=train,
                 val_speakers=val, test_speakers=test)
        f.assert_disjoint()
        folds.append(f)
    return folds


def leave_one_speaker_out(speakers: dict[str, dict],
                          val_fraction: float,
                          seed: int) -> list[Fold]:
    """One fold per speaker — that speaker is the test set."""
    folds = []
    rng = random.Random(seed)
    all_spk = sorted(speakers.keys())
    for i, held_out in enumerate(all_spk):
        remaining = [s for s in all_spk if s != held_out]
        rng2 = random.Random(seed + 1000 + i)
        remaining = remaining[:]
        rng2.shuffle(remaining)
        n_val = max(1, int(round(len(remaining) * val_fraction)))
        val = sorted(remaining[:n_val])
        train = sorted(remaining[n_val:])
        f = Fold(index=i, train_speakers=train,
                 val_speakers=val, test_speakers=[held_out])
        f.assert_disjoint()
        folds.append(f)
    return folds


# ----------------------------------------------------------- top-level driver
def build_manifest(cfg: dict) -> SplitManifest:
    """Discover TORGO, build the split under the configured protocol, return
    a manifest ready to serialize. The manifest is the single source of truth
    consumed by training, baselines and ablations (#14, #29, #32).
    """
    samples = discover_torgo(cfg["dataset"]["root"])
    speakers = speakers_index(samples)

    protocol = cfg["split"]["protocol"]
    seed = int(cfg["split"]["seed"])
    if protocol == "speaker_stratified_kfold":
        folds = speaker_stratified_kfold(
            speakers,
            k=int(cfg["split"]["k_folds"]),
            val_fraction=float(cfg["split"]["val_fraction_of_train"]),
            stratify_on=list(cfg["split"]["stratify_on"]),
            seed=seed,
        )
    elif protocol == "leave_one_speaker_out":
        folds = leave_one_speaker_out(
            speakers,
            val_fraction=float(cfg["split"]["val_fraction_of_train"]),
            seed=seed,
        )
    elif protocol == "sample_level_legacy":
        raise ValueError(
            "sample_level_legacy is OFF by default (#15). "
            "Enable explicitly via `experiment.legacy_sample_split=true` and "
            "run the dedicated legacy script; it is not supported in the main flow."
        )
    else:
        raise ValueError(f"Unknown split protocol: {protocol}")

    manifest = SplitManifest(
        protocol=protocol,
        k_folds=len(folds),
        seed=seed,
        dataset_root=str(cfg["dataset"]["root"]),
        stratify_on=list(cfg["split"]["stratify_on"]),
        folds=folds,
        speakers=speakers,
    )
    return manifest


def samples_for_fold(samples: list[dict], fold: Fold) -> dict[str, list[dict]]:
    """Partition the sample list into {train, val, test} by speaker membership."""
    tr = set(fold.train_speakers)
    va = set(fold.val_speakers)
    te = set(fold.test_speakers)
    out = {"train": [], "val": [], "test": []}
    for s in samples:
        spk = s["speaker"]
        if spk in tr:
            out["train"].append(s)
        elif spk in va:
            out["val"].append(s)
        elif spk in te:
            out["test"].append(s)
        # else: speaker not in this fold at all (shouldn't happen)
    return out


# ===================================================================== quality
# --------------------------------------------------------------- quality report
def fold_quality(samples: list[dict], fold: Fold) -> dict:
    """Compute speaker / sample / label / gender counts for one fold."""
    sbs = samples_for_fold(samples, fold)
    rec: dict = {"fold": fold.index}
    for split, subset in sbs.items():
        spk_set = {s["speaker"] for s in subset}
        pos = sum(1 for s in subset if s["label"] == 1)
        neg = sum(1 for s in subset if s["label"] == 0)
        female = sum(1 for s in subset if s["gender"] == "F")
        male = sum(1 for s in subset if s["gender"] == "M")
        rec.update({
            f"{split}_speakers": len(spk_set),
            f"{split}_samples":  len(subset),
            f"{split}_pos":      pos,
            f"{split}_neg":      neg,
            f"{split}_female":   female,
            f"{split}_male":     male,
        })
    return rec


class FoldQualityError(AssertionError):
    """Raised when a fold violates a hard quality invariant (empty split,
    single-class test split, etc.).  Caller may catch and downgrade to a
    warning depending on context."""


def validate_fold_quality(rec: dict, *, strict: bool = True,
                          logger=None) -> list[str]:
    """Check a per-fold record against hard / soft invariants.

    Returns a list of human-readable warning strings.
    Raises :class:`FoldQualityError` on hard violations when ``strict=True``.
    """
    issues: list[str] = []
    fold_idx = rec.get("fold", "?")
    for split in ("train", "val", "test"):
        if rec[f"{split}_samples"] == 0:
            msg = f"fold {fold_idx}: {split} split is empty"
            if strict:
                raise FoldQualityError(msg)
            issues.append(msg)
    if rec["test_pos"] == 0 or rec["test_neg"] == 0:
        msg = (f"fold {fold_idx}: test has only one class "
               f"(pos={rec['test_pos']}, neg={rec['test_neg']})")
        if strict:
            raise FoldQualityError(msg)
        issues.append(msg)
    # val single-class is a soft warning (permissible with very small datasets)
    if rec["val_pos"] == 0 or rec["val_neg"] == 0:
        msg = (f"fold {fold_idx}: val has only one class "
               f"(pos={rec['val_pos']}, neg={rec['val_neg']})")
        issues.append(msg)
    for msg in issues:
        if logger is not None:
            logger.warning(msg)
    return issues


def build_quality_report(samples: list[dict],
                         manifest: "SplitManifest",
                         *, strict: bool = True,
                         logger=None) -> list[dict]:
    """Build one quality record per fold. Validates on the fly."""
    records = []
    for fold in manifest.folds:
        rec = fold_quality(samples, fold)
        validate_fold_quality(rec, strict=strict, logger=logger)
        records.append(rec)
    return records


def write_quality_report(records: list[dict], out_dir) -> tuple:
    """Write the quality report next to the manifest.  Returns paths."""
    from pathlib import Path
    import json as _json
    import pandas as pd
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "split_quality.json"
    csv_path = out_dir / "split_quality.csv"
    with open(json_path, "w") as f:
        _json.dump(records, f, indent=2)
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return json_path, csv_path


def format_quality_table(records: list[dict]) -> str:
    """Short terminal-friendly view."""
    if not records:
        return "(no folds)"
    lines = []
    header = (f"{'fold':>4} | "
              f"{'tr_spk':>6} {'va_spk':>6} {'te_spk':>6} | "
              f"{'tr_N':>5} {'va_N':>5} {'te_N':>5} | "
              f"{'te_pos':>6} {'te_neg':>6} | "
              f"{'te_F':>5} {'te_M':>5}")
    lines.append(header)
    lines.append("-" * len(header))
    for r in records:
        lines.append(
            f"{r['fold']:>4} | "
            f"{r['train_speakers']:>6} {r['val_speakers']:>6} {r['test_speakers']:>6} | "
            f"{r['train_samples']:>5} {r['val_samples']:>5} {r['test_samples']:>5} | "
            f"{r['test_pos']:>6} {r['test_neg']:>6} | "
            f"{r['test_female']:>5} {r['test_male']:>5}"
        )
    return "\n".join(lines)
