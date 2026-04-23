"""Train the Brivit main model for one (seed, fold) combination.

Returns a metric row compatible with baseline rows so everything flows into
the same aggregator and statistics pipeline.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from brivit.augment.audio_augment import build_from_config as build_augment
from brivit.data.dataset import make_loaders
from brivit.data.splits import samples_for_fold
from brivit.evaluation.metrics import core_metrics, evaluate_arrays
from brivit.models.brivit import build_from_config as build_model
from brivit.preprocess.brain_inspired import build_from_config as build_preproc
from brivit.training.optim import build as build_opt
from brivit.utils.logging_utils import make_logger, set_seed


def _predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, ss = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            s = torch.softmax(model(x), -1)[:, 1].cpu().numpy()
            ys.append(y.numpy()); ss.append(s)
    y_true = np.concatenate(ys)
    y_score = np.concatenate(ss)
    y_pred = (y_score >= 0.5).astype(int)
    return y_true, y_pred, y_score


def _train_one_core(cfg: dict, samples: list[dict], fold,
                    seed: int, run_paths, logger=None):
    """Internal: run one (seed, fold) and return raw (y_true, y_pred, y_score).

    Callers that want a metric row use :func:`train_one`; callers that want
    to combine predictions across sibling models (e.g. gender ablation)
    use :func:`train_one_return_preds`.
    """
    logger = logger or make_logger("brivit.train", run_paths.logs / "train.log")
    set_seed(seed, cfg["seeds"]["deterministic"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data.  Preprocessor stays on CPU — it must not be touched by CUDA
    # because DataLoader workers need a fork-safe, CUDA-free object.
    # Training loop moves each batch tensor to `device` instead (revision #1).
    sbs = samples_for_fold(samples, fold)
    preproc = build_preproc(cfg)
    preproc.eval()
    for p in preproc.parameters():
        p.requires_grad_(False)
    wave_aug, spec_aug = build_augment(cfg)
    loaders = make_loaders(
        sbs, preproc,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        sample_rate=cfg["audio"]["sampling_rate"],
        clip_s=cfg["audio"]["clip_duration_s"],
        waveform_aug=wave_aug, spec_aug=spec_aug,
    )

    # Model
    model = build_model(cfg).to(device)
    optimizer = build_opt(cfg["train"]["optimizer"], model.parameters(),
                          cfg["train"]["learning_rate"],
                          cfg["train"]["weight_decay"])
    ce = nn.CrossEntropyLoss()

    # Early stopping on val_auc
    best_auc = -1.0; best_state = None; patience = 0
    es = cfg["train"]["early_stopping"]

    for ep in range(cfg["train"]["epochs"]):
        model.train()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            optimizer.step()

        y_t, y_p, y_s = _predict(model, loaders["val"], device)
        m = core_metrics(y_t, y_p, y_s)
        logger.info(f"[fold={fold.index} seed={seed} ep={ep}] "
                    f"val_auc={m['roc_auc']:.4f} val_acc={m['accuracy']:.4f}")

        if m["roc_auc"] > best_auc:
            best_auc = m["roc_auc"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if es["enabled"] and patience >= es["patience"]:
                logger.info(
                    f"[fold={fold.index} seed={seed}] early stop at ep {ep}"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        ckpt = (run_paths.checkpoints
                / f"brivit_fold{fold.index}_seed{seed}.pt")
        torch.save(best_state, ckpt)

    # Final test evaluation on ORIGINAL (non-augmented) test split
    y_t, y_p, y_s = _predict(model, loaders["test"], device)
    return y_t, y_p, y_s


def train_one_return_preds(cfg: dict, samples: list[dict], fold,
                           seed: int, run_paths, logger=None):
    """Public variant that returns (y_true, y_pred, y_score) arrays.

    Used by pipelines that want to combine predictions from multiple sibling
    models before computing metrics — notably the gender-aware ablation
    branch (revision #2).
    """
    return _train_one_core(cfg, samples, fold, seed, run_paths, logger)


def _metrics_row(cfg, y_true, y_pred, y_score, model_name, seed, fold_index):
    row = evaluate_arrays(
        y_true, y_pred, y_score,
        n_bins=cfg["evaluation"]["ece_bins"],
        bootstrap_iters=cfg["evaluation"]["bootstrap_iters"],
        seed=seed,
    )
    row.update({
        "model": model_name,
        "seed": seed,
        "fold": fold_index,
        "subject_disjoint": True,
    })
    return row


def train_one(cfg: dict, samples: list[dict], fold,
              seed: int, run_paths, logger=None) -> dict:
    """Train one (seed, fold) and return a metric row."""
    y_t, y_p, y_s = _train_one_core(cfg, samples, fold, seed, run_paths, logger)
    return _metrics_row(cfg, y_t, y_p, y_s, "Brivit", seed, fold.index)
