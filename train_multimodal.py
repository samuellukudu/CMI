"""K-Fold training loop for multimodal gesture classification (Stage-3).

Trains `MultiModalTransformerClassifier` on the 8 BFRB gestures using the
cross-validation splits provided in ``preprocessed/cv_splits.joblib``.
At the end of each fold, performance is measured with the official
`CompetitionMetric` (hierarchical F1) defined in ``cmi_2025.py``.
"""
from __future__ import annotations

import argparse
import gc
import os
import random
from pathlib import Path
from typing import Tuple, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm, trange

from multimodal_dataset import MultiSensorDataset
from multimodal_model import MultiModalTransformerClassifier, MultiModalConfig
from cmi_2025 import CompetitionMetric
from utils import seeding, flush

# -----------------------------------------------------------------------------
#  Reproducibility & helpers
# -----------------------------------------------------------------------------

ROOT = Path(__file__).parent
PREP = ROOT / "preprocessed"


# -----------------------------------------------------------------------------
#  Data loading
# -----------------------------------------------------------------------------

def load_artifacts() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """Load preprocessed arrays and CV splits for BFRB gestures."""
    imu = np.load(PREP / "train_imu.npy")
    thm = np.load(PREP / "train_thm.npy")
    tof = np.load(PREP / "train_tof.npy")

    cv = joblib.load(PREP / "cv_splits.joblib")
    y_str = cv["bfrb_targets"]  # array[str] length N_bfrb
    splits = cv["bfrb_splits"]  # list of (train_idx, val_idx)

    # Ensure sensor arrays align with BFRB-only target rows.
    target_mask = cv["binary_targets"] == 1  # True for Target sequences

    assert target_mask.sum() == len(y_str), (
        "Mismatch between number of Target rows in mask and bfrb_targets length"
    )

    imu = imu[target_mask]
    thm = thm[target_mask]
    tof = tof[target_mask]

    return imu, thm, tof, y_str, splits


def build_label_mapping(labels: np.ndarray) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Map gesture strings to integer class indices (sorted for stability)."""
    uniq = sorted(np.unique(labels))
    str2idx = {g: i for i, g in enumerate(uniq)}
    idx2str = {i: g for g, i in str2idx.items()}
    return str2idx, idx2str


# -----------------------------------------------------------------------------
#  Training utilities
# -----------------------------------------------------------------------------

def evaluate(model: MultiModalTransformerClassifier, loader: DataLoader, idx2str: Dict[int, str], device: str) -> float:
    """Return hierarchical F1 using CompetitionMetric on *loader*."""
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for imu, thm, tof, y in loader:
            imu, thm, tof = imu.to(device), thm.to(device), tof.to(device)
            logits = model(imu, thm, tof)
            pred_idx = logits.argmax(dim=1).cpu().numpy()
            true_idx = y.cpu().numpy()
            all_preds.extend(pred_idx)
            all_trues.extend(true_idx)

    # Convert indices to gesture strings
    preds_str = [idx2str[i] for i in all_preds]
    trues_str = [idx2str[i] for i in all_trues]

    df_pred = pd.DataFrame({"gesture": preds_str})
    df_true = pd.DataFrame({"gesture": trues_str})

    metric = CompetitionMetric()
    f1 = metric.calculate_hierarchical_f1(df_true, df_pred)
    return f1


def train_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    imu: np.ndarray,
    thm: np.ndarray,
    tof: np.ndarray,
    y_idx: np.ndarray,
    cfg: MultiModalConfig,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    idx2str: Dict[int, str],
) -> float:
    # Datasets & loaders
    ds = MultiSensorDataset(imu, thm, tof, y_idx)
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False)

    model = MultiModalTransformerClassifier(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_f1 = -1.0

    for epoch in trange(epochs, desc=f"Fold {fold}"):
        model.train()
        running_loss = 0.0
        for imu_b, thm_b, tof_b, y_b in tqdm(train_loader, desc="train", leave=False):
            imu_b, thm_b, tof_b = imu_b.to(device), thm_b.to(device), tof_b.to(device)
            y_b = y_b.long().to(device)
            opt.zero_grad()
            logits = model(imu_b, thm_b, tof_b)
            loss = criterion(logits, y_b)
            loss.backward()
            opt.step()
            running_loss += loss.item() * imu_b.size(0)
        avg_loss = running_loss / len(train_loader.dataset)

        # Validation metric
        val_f1 = evaluate(model, val_loader, idx2str, device)
        tqdm.write(f"Fold {fold} Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} val_hier_f1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            # Save checkpoint
            os.makedirs(ROOT / "checkpoints", exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "val_hier_f1": val_f1,
            }, ROOT / f"checkpoints/multimodal_fold{fold}.pt")

    return best_f1


# -----------------------------------------------------------------------------
#  Main entry
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    if torch.cuda.is_available():
        default_device = "cuda"
    elif torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seeding(args.seed)

    print("Loading preprocessed data & CV splits…")
    imu, thm, tof, y_str, splits = load_artifacts()
    str2idx, idx2str = build_label_mapping(y_str)
    num_classes = len(str2idx)
    y_idx = np.vectorize(str2idx.get)(y_str)

    cfg = MultiModalConfig(num_classes=num_classes)
    print(f"Model configuration: {cfg}")

    best_scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n======== Training Fold {fold+1}/{len(splits)} ========")
        best_f1 = train_fold(
            fold,
            train_idx,
            val_idx,
            imu,
            thm,
            tof,
            y_idx,
            cfg,
            args.epochs,
            args.batch_size,
            args.lr,
            args.device,
            idx2str,
        )
        best_scores.append(best_f1)
        flush()

    print("\n=== Cross-validated Hierarchical F1 ===")
    for i, s in enumerate(best_scores):
        print(f"Fold {i}: {s:.4f}")
    print(f"Mean F1: {np.mean(best_scores):.4f}")


if __name__ == "__main__":
    main()