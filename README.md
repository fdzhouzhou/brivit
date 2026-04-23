# Brivit — TORGO subject-disjoint reproduction

Public reproduction of the Brivit framework
(_Audio-based Risk-related Neurological Assessment via Brain-inspired Data
Engineering_) on the **TORGO** dysarthric speech dataset.

> **Scope.** Only what is needed to reproduce the revised paper. All
> nursing-home data, commercial deployment modules, service endpoints, and
> private dependencies have been removed. Main results are always
> **subject-disjoint on TORGO**.

---

## 1. Repository layout

```
project_root/
├── README.md
├── pyproject.toml
├── requirements-cpu.txt
├── requirements-gpu.txt
├── configs/
│   └── default.yaml
├── brivit/                         ← the installable Python package
│   ├── __init__.py
│   ├── augment/       audio_augment.py
│   ├── baselines/     cnn.py  features.py  runners.py
│   ├── config/        loader.py
│   ├── data/          dataset.py  splits.py
│   ├── evaluation/    metrics.py  aggregate.py  significance.py
│   ├── models/        brivit.py
│   ├── preprocess/    brain_inspired.py
│   ├── training/      optim.py  train_main.py  train_baseline.py
│   └── utils/         logging_utils.py  runs.py
├── scripts/
│   ├── build_split.py
│   ├── run_main.py
│   ├── run_baselines.py
│   ├── run_ablation.py
│   └── summarize.py
└── tests/
    ├── test_splits.py         (unit, CPU-only)
    ├── test_aggregate.py      (unit, CPU-only)
    ├── test_metrics.py        (unit, CPU-only)
    ├── test_dataset.py        (integration, needs torch)
    └── test_model.py          (integration, needs torch)
```

Everything the user runs is launched from this single root. No zip-in-zip,
no duplicate README / configs.

---

## 2. Installation

**Python range:** `3.9 ≤ python < 3.12`
**Pinned:** `torch == 2.2.2`, `torchaudio == 2.2.2`. They must match.

### CPU-only install

```bash
pip install -e .[test-cpu] --extra-index-url https://download.pytorch.org/whl/cpu
```

Or, without editable install:

```bash
pip install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### GPU install (CUDA 12.1)

```bash
pip install -e .[test-gpu] --extra-index-url https://download.pytorch.org/whl/cu121
```

Or:

```bash
pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Testing

The suite is split into three marker groups:

| Marker | What it covers | Needs |
|---|---|---|
| `unit`        | splits, aggregation, metrics/CI, config | Python only — never imports torch |
| `integration` | dataset contract, model forward API | torch + torchaudio (CPU sufficient) |
| `gpu`         | CUDA-specific tests                    | CUDA runtime |

`pytest -q` runs `unit` + `integration` (excludes `gpu` by default — see
`pyproject.toml::[tool.pytest.ini_options]`).

### CPU environment

```bash
pip install -e .[test-cpu] --extra-index-url https://download.pytorch.org/whl/cpu
pytest -q -m "not gpu"
```

If `torch` isn't available in the env for any reason, the integration tests
auto-skip instead of failing, so `pytest -q -m unit` always succeeds without
torch.

### GPU environment

```bash
pip install -e .[test-gpu] --extra-index-url https://download.pytorch.org/whl/cu121
pytest -q
```

---

## 4. Data preparation (TORGO only)

Download TORGO from
<https://www.kaggle.com/datasets/pranaykoppula/torgo-audio>
and place it as:

```
data/TORGO/
├── F_Con/  FC01/ Session1/ wav_headMic/*.wav …
├── F_Dys/  F01/  Session1/ …
├── M_Con/  MC01/ …
└── M_Dys/  M01/  …
```

Speaker IDs are parsed from folder names (`F01`, `FC02`, `M03`, `MC04`, …).
The `C` infix marks a control speaker (label 0); otherwise label 1.

---

## 5. One-time split build

```bash
python -m scripts.build_split --config configs/default.yaml
```

Writes, at the manifest's parent directory:

| File | Purpose |
|---|---|
| `split_manifest.json` | speaker lists per fold (consumed by every other script) |
| `split_quality.json` / `.csv` | speakers / samples / pos & neg counts / female & male counts per split per fold |

At build time:

* pairwise speaker-disjoint assertions run for every fold;
* an **empty** train / val / test split raises `FoldQualityError`;
* a **single-class** test fold (only positives or only negatives) raises `FoldQualityError`;
* a single-class val fold emits a warning (not hard-blocked; rare on small data).

`--no-strict` downgrades the hard assertions to warnings. Do not use it for
paper runs.

A compact per-fold table is printed on stdout so imbalances are visible
without opening the CSV:

```
fold | tr_spk va_spk te_spk | tr_N  va_N  te_N  | te_pos te_neg | te_F  te_M
-----+------------------------+--------------------+---------------+-----------
   0 |     9      3      3   |  N1    N2    N3    |     P      Q   |   F    M
   ...
```

---

## 6. Experiments

```bash
# Main model: 5 seeds × 5 folds
python -m scripts.run_main       --config configs/default.yaml

# All baselines on the SAME split
python -m scripts.run_baselines  --config configs/default.yaml

# Ablation: six switches, each flipped independently
python -m scripts.run_ablation   --config configs/default.yaml

# Combine main + baselines → paper-ready table + paired Wilcoxon
python -m scripts.summarize \
    --main_dir runs/brivit_torgo_subject_disjoint/<ts>/results \
    --base_dir runs/brivit_torgo_subject_disjoint__baselines/<ts>/results
```

Subset flags and CLI overrides:

```bash
python -m scripts.run_main       --folds 0 1 --seeds 0 1
python -m scripts.run_baselines  --models svm random_forest
python -m scripts.run_ablation   --only use_dual_channel_attn use_augment
python -m scripts.run_main       --override train.epochs=30 train.batch_size=8
```

---

## 7. Outputs

Every script creates `runs/<experiment>/<timestamp>/`:

```
config.snapshot.yaml               exact config used
split_manifest.json                speaker lists for every fold
split_quality.json / .csv          per-fold speaker / sample / label / gender counts
meta.json                          seeds, folds, model list, counts
runtime_meta.json                  env snapshot — python, torch, cuda,
                                   git commit, config/manifest hashes
checkpoints/                       best-validation-AUC models
logs/                              structured text logs
results/
├── main_results_per_run.csv       one row per (seed, fold)
├── main_results_summary.csv       mean / std / 95 % CI per model
├── baselines_per_run.csv          identical schema to main
├── baselines_summary.csv
├── ablation_per_run.csv           grouped by switch
├── ablation_summary.csv
└── ci_methods.json                how each CI layer was computed
```

`scripts.summarize` additionally writes:

```
paper_main_table.csv  /  .json     drop into the paper directly
statistics.csv        /  .json     paired Wilcoxon + Holm correction
runtime_meta.json                  summarizer's own environment snapshot
```

### Two CI layers — kept separate on purpose

The 8 core metrics —
`accuracy, precision, recall, specificity, f1, balanced_accuracy, roc_auc, pr_auc`
— each appear with **per-run** and **summary** CI columns that are
_named differently_ so readers cannot accidentally mix them:

| Layer | Column suffix | Method | What it measures |
|---|---|---|---|
| per-run | `<metric>_ci_lo`, `<metric>_ci_hi` | **bootstrap** (1000 resamples) | uncertainty inside ONE test split (one seed, one fold) |
| summary | `<metric>_summary_ci95_lo`, `<metric>_summary_ci95_hi` | **normal approximation** (mean ± 1.96·SEM) | variance ACROSS (seed, fold) point estimates |

Each results directory also contains `ci_methods.json`:

```json
{
  "per_run_ci":  {"method": "bootstrap",              ...},
  "summary_ci":  {"method": "normal_approximation",   ...}
}
```

No column in any summary table will be labelled just `_ci95_lo` / `_ci95_hi`
— that would be ambiguous. All bootstrap code goes through a single shared
loop (`bootstrap_ci_all`), so adding or removing a metric does not require
editing multiple places.

### Invariants enforced in code (not just convention)

* `TorgoDataset` raises if you try to attach augmentation to `val` or `test`.
* `scripts.build_split` runs pairwise speaker-disjoint assertions per fold
  AND raises `FoldQualityError` if any fold has an empty split or a
  single-class test set.
* `evaluation/aggregate.py::write_main_results` refuses rows where
  `subject_disjoint != True`.
* `scripts.run_main` refuses to start if `experiment.legacy_sample_split=true`.
* Preprocessor is **never** moved to CUDA; it lives on CPU and is safe to
  pickle for DataLoader workers.

---

## 8. Model: attention and feature export

The model's default forward returns `logits` — a plain Tensor. The training
hot path is unaffected.

Opt-in structured output:

```python
from brivit.models.brivit import Brivit, ViTConfig

model = Brivit(ViTConfig(...))
out = model(x, return_dict=True)
assert "logits" in out        # (B, num_classes)
assert "features" in out      # (B, embed_dim)   — pre-head CLS vector
assert "attention" in out     # last encoder layer's attention
```

### Attention shape

Each attention Tensor has shape:

```
(B, heads, N, N)
    where
        B     = batch size
        heads = cfg.num_heads              (default 8)
        N     = n_patches + 1              (patches + CLS token)
                = (img_size // patch_size)^2 + 1
                = (224 / 32)^2 + 1 = 50
```

### Attention keys

For the **dual-channel** model (paper's default, `use_dual_channel_attn=True`):

```
out["attention"] = {
    "channel_a": Tensor (B, heads, N, N),
    "channel_b": Tensor (B, heads, N, N),
    "fused":     Tensor (B, heads, N, N),   # elementwise mean of A and B
}
```

For the **single-path** ablation (`use_dual_channel_attn=False`):

```
out["attention"] = {"single": Tensor (B, heads, N, N)}
```

### Selective flags and all-layer attention

```python
out = model(x, return_attention=True)       # dict, attention from last layer
out = model(x, return_features=True)        # dict, only features
out = model(x, return_dict=True,
            all_layers=True)                # also "attention_all_layers"
                                            # (list of per-layer dicts)
```

Default call `model(x)` is unchanged: a bare Tensor, no dict wrapping,
no attention-weight computation overhead.

---

## 9. Configuration cheatsheet

All parameters live in `configs/default.yaml`. Changeable from the CLI via
`--override key.path=value`. Most common overrides:

| Section | Key | Default |
|---|---|---|
| `split`   | `protocol` / `k_folds` / `stratify_on`          | `speaker_stratified_kfold` / `5` / `[label, gender]` |
| `train`   | `optimizer` / `learning_rate` / `batch_size` / `epochs` / `weight_decay` | `RectifiedAdam` / `1e-4` / `16` / `100` / `1e-4` |
| `seeds`   | `model_init`                                    | `[0, 1, 2, 3, 4]` |
| `ablation`| six switches, all `true` by default             | see `configs/default.yaml` |
| `model`   | `patch_size` / `embed_dim` / `depth` / `num_heads` / `drop_path_rate` | `32 / 256 / 6 / 8 / 0.1` |
| `audio`   | `n_mels` / `n_fft` / `hop_length` / `sampling_rate` | `64 / 400 / 160 / 16000` |
| `spike_encoding` | `tau_m_ms` / `v_threshold` / `tau_ref_ms` / `dt_ms` / `membrane_decay` / `event_threshold` / `density_percentile` | `20 / 1.0 / 5 / 1 / 0.95 / 0.2 / 75` |
| `baselines.threshold` | `strategy` / `target_sensitivity` | `max_f1` / `0.90` |

### Semantics notes

**`use_gender_branch` ablation** — Produces ONE row per `(seed, fold)`: a
female-only and a male-only model are trained independently on the same
config, their test predictions are concatenated, and metrics are computed
once on the combined set. The table distinguishes `full_model` (unified)
from `gender_aware_branch` (this variant) — not from per-gender rows.

**Classical baselines (SVM / RF) use validation** — `train → fit`,
`val → pick decision threshold` (config-driven: `max_f1`,
`max_balanced_accuracy`, or `fixed_sensitivity` with a configurable
`target_sensitivity`), `test → evaluate at that frozen threshold`. The
chosen threshold and the selection strategy are saved on every baseline
result row for audit.

**`spike_encoding.event_threshold`** — The LIF spike-density mask keeps a
spike only if its local firing rate passes BOTH the per-channel
`density_percentile` (relative) AND the absolute `event_threshold`.
Changing the config value does move the output features.

**Preprocessor placement** — The brain-inspired preprocessor lives on CPU
throughout (it is frozen at build time). DataLoader workers can fork it
safely because it holds no CUDA handles. Only the resulting feature map
is moved to `device` inside the training loop.

---

## 10. Full reproduction walkthrough

```bash
# install
pip install -e .[test-cpu] --extra-index-url https://download.pytorch.org/whl/cpu

# sanity
pytest -q                                          # unit + integration pass

# one-off: fix the split
python -m scripts.build_split --config configs/default.yaml

# experiments
python -m scripts.run_main       --config configs/default.yaml
python -m scripts.run_baselines  --config configs/default.yaml
python -m scripts.run_ablation   --config configs/default.yaml

# paper table
python -m scripts.summarize \
    --main_dir runs/brivit_torgo_subject_disjoint/<ts>/results \
    --base_dir runs/brivit_torgo_subject_disjoint__baselines/<ts>/results \
    --out_dir  runs/summary
```

The summary answers the paper's revised headline question directly:
**how well does Brivit discriminate dysarthric from control speech on
speakers it has never heard?** Nothing in this pipeline can leak speakers
across train / val / test, and no result row that isn't subject-disjoint can
be written to the main table.

---

## 11. Version notes

* `0.4.0` — Preprocessor stays on CPU (DataLoader-worker safe).
  `use_gender_branch` ablation now emits ONE unified row per (seed, fold)
  by concatenating the per-gender test predictions. `event_threshold` is
  actually wired into the spike-density mask. SVM / RF baselines pick
  decision threshold on the validation split before scoring test. Summary
  CI columns renamed to `<metric>_summary_ci95_{lo,hi}` to disambiguate
  from per-run bootstrap CI; every results directory ships a
  `ci_methods.json` sidecar. `build_split` emits a per-fold quality report
  (speakers / samples / label / gender counts) and blocks empty or
  single-class folds by default. Every run writes a `runtime_meta.json`
  (python / torch / cuda / git / config hash / manifest hash). New
  self-check tests for LOSO uniqueness, fold quality assertions,
  dual-vs-single output shape, and runtime-meta contents.

* `0.3.0` — Install split into CPU / GPU extras (`.[cpu]` / `.[gpu]`);
  torch and torchaudio pinned to `2.2.2`. `torchaudio` moved to lazy imports
  so unit tests never trigger CUDA loader lookups. `pytest` marker groups
  (`unit` / `integration` / `gpu`). `model.forward(..., return_dict=True)`
  provides standardized access to logits, features and per-channel attention.
  Bootstrap CI now covers all 8 core metrics via a single shared loop.

* `0.2.0` — TORGO-only main flow, subject-disjoint splits, baselines,
  ablation driver, significance tests, unified result schema. Removed:
  nursing-home data, sample-level splits as default, deployment modules.

* `0.1.x` — Initial reproduction (deprecated; see git history).
