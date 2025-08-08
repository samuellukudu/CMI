"""K-Fold training loop for binary BFRB detection with progress bars.

Loads preprocessed IMU features + CV splits, trains BinaryIMUCNN on each fold,
prints Binary F1 after each validation.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm, trange

from imu_dataset import BinaryIMUDataset
from imu_model import BinaryIMUCNN, BinaryIMUConfig

ROOT = Path(__file__).parent
PREP = ROOT / "preprocessed"


def load_artifacts() -> Tuple[np.ndarray, np.ndarray, list]:
    imu = np.load(PREP / "train_imu.npy")  # (N, F)
    cv = joblib.load(PREP / "cv_splits.joblib")
    y = cv["binary_targets"]  # (N,)
    splits = cv["binary_splits"]
    return imu, y, splits


def train_fold(fold: int, train_idx: np.ndarray, val_idx: np.ndarray, imu: np.ndarray, y: np.ndarray, cfg: BinaryIMUConfig, epochs: int, batch_size: int, lr: float, device: str) -> float:
    # Dataset & loaders
    ds = BinaryIMUDataset(imu, y)
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False)

    model = BinaryIMUCNN(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in trange(epochs, desc=f"Fold {fold}"):
        model.train()
        running_loss = 0.0
        for xb, yb in tqdm(train_loader, desc="train", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = model.loss(logits, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item() * xb.size(0)
        avg_loss = running_loss / len(train_loader.dataset)

        # --- validation ---
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb).squeeze(-1)
                val_loss_total += model.loss(logits, yb).item() * xb.size(0)
        avg_val_loss = val_loss_total / len(val_loader.dataset)

        tqdm.write(f"Fold {fold} Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} val_loss: {avg_val_loss:.4f}")

    # Evaluation
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in tqdm(val_loader, desc="val", leave=False):
            xb = xb.to(device)
            logits = model(xb).squeeze(-1).cpu()
            preds.append((logits > 0.0).int())  # threshold 0.5 on sigmoid equivalent
            targets.append(yb.int())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    f1 = f1_score(targets, preds)
    tqdm.write(f"Fold {fold} Binary F1: {f1:.4f}\n")
    return f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    imu, y, splits = load_artifacts()
    cfg = BinaryIMUConfig(in_channels=imu.shape[1])

    f1_scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        f1 = train_fold(fold, train_idx, val_idx, imu, y, cfg, args.epochs, args.batch_size, args.lr, args.device)
        f1_scores.append(f1)

    print("\n=== Cross-validated Binary F1 ===")
    for i, s in enumerate(f1_scores):
        print(f"Fold {i}: {s:.4f}")
    print(f"Mean F1: {np.mean(f1_scores):.4f}")


if __name__ == "__main__":
    main()