"""Unified baseline train + evaluate runners.

All baselines:
    * Consume the SAME TORGO subject-disjoint split as the main model (#29).
    * Return rows with the exact same metric columns as the main model (#30).
    * Never see augmented samples at eval time; training may use raw features
      but NOT speech augmentation (a fair ablation is that augmentation is
      a property of the main method, not of feature engineering baselines).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset

from brivit.baselines.cnn import CNN2D, CRNN
from brivit.baselines.features import (ClassicalFeatureConfig, LogMelConfig,
                                       build_logmel_tensor, extract_classical)
from brivit.evaluation.metrics import evaluate_arrays


# =================================================================== sklearn
def _scores_from_pipe(pipe, X: np.ndarray) -> np.ndarray:
    """Map whatever the fitted pipeline exposes to [0, 1] scores."""
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    df = pipe.decision_function(X)
    return (df - df.min()) / (df.max() - df.min() + 1e-8)


def _select_threshold(y_val: np.ndarray, scores_val: np.ndarray,
                      strategy: str, target_sensitivity: float) -> float:
    """Pick a decision threshold on the validation split.

    Supported strategies (revision #4):
        max_f1                — threshold maximising F1
        max_balanced_accuracy — threshold maximising balanced accuracy
        fixed_sensitivity     — highest threshold whose recall >= target
                                (falls back to max_balanced_accuracy if the
                                target cannot be met).
    """
    from sklearn.metrics import (balanced_accuracy_score, f1_score,
                                 recall_score)

    # Dense grid over unique score values plus a small margin
    grid = np.unique(np.concatenate([scores_val, [0.0, 1.0]]))
    if grid.size < 2:
        return 0.5
    candidates = (grid[:-1] + grid[1:]) / 2.0
    candidates = np.concatenate([candidates, [0.5]])

    if strategy == "max_f1":
        scores = [f1_score(y_val, (scores_val >= t).astype(int),
                           zero_division=0) for t in candidates]
        return float(candidates[int(np.argmax(scores))])
    if strategy == "max_balanced_accuracy":
        scores = [balanced_accuracy_score(y_val, (scores_val >= t).astype(int))
                  for t in candidates]
        return float(candidates[int(np.argmax(scores))])
    if strategy == "fixed_sensitivity":
        feasible = [(t, balanced_accuracy_score(
                        y_val, (scores_val >= t).astype(int)))
                    for t in candidates
                    if recall_score(y_val, (scores_val >= t).astype(int),
                                    zero_division=0) >= target_sensitivity]
        if feasible:
            return float(max(feasible, key=lambda kv: kv[1])[0])
        # Target infeasible — fall back to balanced accuracy maximization.
        return _select_threshold(y_val, scores_val,
                                 "max_balanced_accuracy",
                                 target_sensitivity)
    raise ValueError(f"Unknown threshold strategy: {strategy!r}")


def run_sklearn_baseline(name: str, clf, samples_by_split: dict, cfg: dict,
                         seed: int) -> dict:
    """Classical baseline with a three-stage protocol:
        * train:  fit the pipeline
        * val:    select decision threshold (policy from config)
        * test:   evaluate with that frozen threshold (revision #4).
    """
    fcfg = ClassicalFeatureConfig(
        sr=cfg["audio"]["sampling_rate"],
        clip_s=cfg["audio"]["clip_duration_s"],
        n_fft=cfg["audio"]["n_fft"],
        win_length=cfg["audio"]["win_length"],
        hop_length=cfg["audio"]["hop_length"],
        n_mels=cfg["audio"]["n_mels"],
    )
    X_tr, y_tr = extract_classical(samples_by_split["train"], fcfg)
    X_va, y_va = extract_classical(samples_by_split["val"], fcfg)
    X_te, y_te = extract_classical(samples_by_split["test"], fcfg)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_tr, y_tr)

    scores_va = _scores_from_pipe(pipe, X_va)
    thr_cfg = cfg["baselines"].get("threshold", {})
    strategy = thr_cfg.get("strategy", "max_f1")
    target = float(thr_cfg.get("target_sensitivity", 0.9))
    chosen_thr = _select_threshold(y_va, scores_va, strategy, target)

    scores_te = _scores_from_pipe(pipe, X_te)
    y_pred = (scores_te >= chosen_thr).astype(int)

    row = evaluate_arrays(
        y_te, y_pred, scores_te,
        n_bins=cfg["evaluation"]["ece_bins"],
        bootstrap_iters=cfg["evaluation"]["bootstrap_iters"],
        seed=seed,
    )
    row["model"] = name
    # Persist the threshold + selection policy for audit (revision #4 requires
    # the chosen threshold to be saved).
    row["threshold"] = chosen_thr
    row["threshold_strategy"] = strategy
    return row


def svm_baseline(samples_by_split, cfg, seed):
    bcfg = cfg["baselines"]["svm"]
    clf = SVC(kernel=bcfg["kernel"], C=bcfg["C"],
              class_weight=bcfg["class_weight"],
              probability=True, random_state=seed)
    return run_sklearn_baseline("SVM", clf, samples_by_split, cfg, seed)


def rf_baseline(samples_by_split, cfg, seed):
    bcfg = cfg["baselines"]["random_forest"]
    clf = RandomForestClassifier(
        n_estimators=bcfg["n_estimators"],
        max_depth=bcfg["max_depth"],
        class_weight=bcfg["class_weight"],
        random_state=seed, n_jobs=-1,
    )
    return run_sklearn_baseline("RandomForest", clf, samples_by_split, cfg, seed)


# =================================================================== CNN
class _LogmelDataset(Dataset):
    def __init__(self, samples, cfg: LogMelConfig):
        self.samples = samples
        self.cfg = cfg

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x = build_logmel_tensor(self.samples[i], self.cfg)
        return x.float(), int(self.samples[i]["label"])


def _cnn_train(model: nn.Module, loader_tr: DataLoader, loader_va: DataLoader,
               epochs: int, lr: float, device) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    best_auc = -1.0; best_state = None
    from brivit.evaluation.metrics import core_metrics
    for ep in range(epochs):
        model.train()
        for x, y in loader_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward(); opt.step()
        # quick val AUC
        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for x, y in loader_va:
                x = x.to(device)
                s = torch.softmax(model(x), -1)[:, 1].cpu().numpy()
                scores.append(s); labels.append(y.numpy())
        y_s = np.concatenate(scores); y_t = np.concatenate(labels)
        m = core_metrics(y_t, (y_s >= 0.5).astype(int), y_s)
        if m["roc_auc"] > best_auc:
            best_auc = m["roc_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _cnn_eval(model: nn.Module, loader_te: DataLoader, device, cfg,
              seed: int, name: str) -> dict:
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader_te:
            x = x.to(device)
            s = torch.softmax(model(x), -1)[:, 1].cpu().numpy()
            scores.append(s); labels.append(y.numpy())
    y_s = np.concatenate(scores); y_t = np.concatenate(labels)
    y_p = (y_s >= 0.5).astype(int)
    row = evaluate_arrays(
        y_t, y_p, y_s,
        n_bins=cfg["evaluation"]["ece_bins"],
        bootstrap_iters=cfg["evaluation"]["bootstrap_iters"],
        seed=seed,
    )
    row["model"] = name
    return row


def cnn2d_baseline(samples_by_split, cfg, seed):
    torch.manual_seed(seed)
    lcfg = LogMelConfig(
        sr=cfg["audio"]["sampling_rate"],
        clip_s=cfg["audio"]["clip_duration_s"],
        n_fft=cfg["audio"]["n_fft"],
        win_length=cfg["audio"]["win_length"],
        hop_length=cfg["audio"]["hop_length"],
        n_mels=cfg["audio"]["n_mels"],
    )
    bcfg = cfg["baselines"]["cnn2d"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_ds = _LogmelDataset(samples_by_split["train"], lcfg)
    va_ds = _LogmelDataset(samples_by_split["val"], lcfg)
    te_ds = _LogmelDataset(samples_by_split["test"], lcfg)
    nw = cfg["train"]["num_workers"]
    tr = DataLoader(tr_ds, batch_size=bcfg["batch_size"], shuffle=True, num_workers=nw)
    va = DataLoader(va_ds, batch_size=bcfg["batch_size"], shuffle=False, num_workers=nw)
    te = DataLoader(te_ds, batch_size=bcfg["batch_size"], shuffle=False, num_workers=nw)

    model = CNN2D(channels=tuple(bcfg["channels"]), dropout=bcfg["dropout"]).to(device)
    model = _cnn_train(model, tr, va, bcfg["epochs"], bcfg["lr"], device)
    return _cnn_eval(model, te, device, cfg, seed, "CNN2D")


def crnn_baseline(samples_by_split, cfg, seed):
    torch.manual_seed(seed)
    lcfg = LogMelConfig(
        sr=cfg["audio"]["sampling_rate"], clip_s=cfg["audio"]["clip_duration_s"],
        n_fft=cfg["audio"]["n_fft"], win_length=cfg["audio"]["win_length"],
        hop_length=cfg["audio"]["hop_length"], n_mels=cfg["audio"]["n_mels"],
    )
    bcfg = cfg["baselines"]["crnn"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = DataLoader(_LogmelDataset(samples_by_split["train"], lcfg),
                    batch_size=bcfg["batch_size"], shuffle=True,
                    num_workers=cfg["train"]["num_workers"])
    va = DataLoader(_LogmelDataset(samples_by_split["val"], lcfg),
                    batch_size=bcfg["batch_size"], shuffle=False,
                    num_workers=cfg["train"]["num_workers"])
    te = DataLoader(_LogmelDataset(samples_by_split["test"], lcfg),
                    batch_size=bcfg["batch_size"], shuffle=False,
                    num_workers=cfg["train"]["num_workers"])
    model = CRNN(
        cnn_channels=tuple(bcfg["cnn_channels"]),
        rnn_hidden=bcfg["rnn_hidden"],
    ).to(device)
    model = _cnn_train(model, tr, va, bcfg["epochs"], bcfg["lr"], device)
    return _cnn_eval(model, te, device, cfg, seed, "CRNN")


BASELINE_REGISTRY: dict[str, Callable] = {
    "svm":           svm_baseline,
    "random_forest": rf_baseline,
    "cnn2d":         cnn2d_baseline,
    "crnn":          crnn_baseline,
}
