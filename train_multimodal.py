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

def load_artifacts(task: str = "all") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """Load preprocessed arrays and CV splits.

    task="all"        -> 18-class training (8 BFRB + 10 non-BFRB)
    task="bfrb_only"  -> 8-class BFRB-only training (legacy mode; deprecated)
    """
    imu = np.load(PREP / "train_imu.npy")
    thm = np.load(PREP / "train_thm.npy")
    tof = np.load(PREP / "train_tof.npy")

    cv = joblib.load(PREP / "cv_splits.joblib")
    # Strict requirement: full multi-class setup must be present
    if "all_gestures" not in cv or "all_splits" not in cv:
        raise RuntimeError("cv_splits.joblib must contain 'all_gestures' and 'all_splits' for multimodal training. Re-run preprocessing to generate full multi-class labels.")
    y_all = cv["all_gestures"]
    splits = cv["all_splits"]
    return imu, thm, tof, y_all, splits


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
    # Mixed precision on CUDA only for stability
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
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
    optimizer_type: str = "adam",
    scheduler_type: str = "cosine",
) -> float:
    # Datasets & loaders
    ds = MultiSensorDataset(imu, thm, tof, y_idx)
    # Use multiple workers and pinned memory for faster host-device transfer.
    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = device in {"cuda", "mps"}
    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    model = MultiModalTransformerClassifier(cfg).to(device)
    # Optimizer selection
    if optimizer_type.lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Scheduler selection
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    elif scheduler_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
    else:
        scheduler = None

    best_f1 = -1.0

    for epoch in trange(epochs, desc=f"Fold {fold}"):
        model.train()
        running_loss = 0.0
        # Enable mixed precision (AMP) on CUDA only
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
        for imu_b, thm_b, tof_b, y_b in tqdm(train_loader, desc="train", leave=False):
            imu_b, thm_b, tof_b = imu_b.to(device, non_blocking=True), thm_b.to(device, non_blocking=True), tof_b.to(device, non_blocking=True)
            y_b = y_b.long().to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(imu_b, thm_b, tof_b)
                loss = criterion(logits, y_b)
            if device == "cuda":
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            # Batch-level step for OneCycleLR
            if scheduler is not None and scheduler_type == "onecycle":
                scheduler.step()
            running_loss += loss.item() * imu_b.size(0)
        avg_loss = running_loss / len(train_loader.dataset)

        # Epoch-level scheduler step for non-OneCycle
        if scheduler is not None and scheduler_type != "onecycle":
            scheduler.step()

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
            
        # Release memory after each epoch
        flush()

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
    # Optimizer & scheduler options
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam", help="Optimizer to use")
    parser.add_argument("--scheduler", choices=["none", "cosine", "onecycle"], default="cosine", help="Learning rate scheduler")
    # Task: 8-class BFRB-only vs 9-class (BFRB + non_target)
    parser.add_argument("--task", choices=["bfrb_only", "all"], default="all", help="Training target: 8-class BFRB-only or 9-class including non_target")
    # Performance / model size knobs
    parser.add_argument("--num_workers", type=int, default=0, help="Override DataLoader workers (0=auto)")
    parser.add_argument("--tof_img_size", type=int, default=0, help="Resize TOF input for EfficientNet (e.g., 128). 0 keeps default")
    parser.add_argument("--tof_model_name", type=str, default="efficientnet_b0", help="timm EfficientNet name for TOF branch (e.g., efficientnet_b0)")
    parser.add_argument("--no_freeze_tof_backbone", action="store_true", help="Unfreeze TOF EfficientNet backbone")
    # Alias flag to match naming in train_imu2d for convenience
    parser.add_argument("--no_freeze_backbone", action="store_true", help="Alias for --no_freeze_tof_backbone")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing preprocessed artifacts")
    args = parser.parse_args()

    seeding(args.seed)

    # Override preprocessed data directory if provided
    global PREP
    if args.data_dir is not None:
        PREP = Path(args.data_dir)

    print("Loading preprocessed data & CV splits…")
    imu, thm, tof, y_str, splits = load_artifacts(args.task)
    str2idx, idx2str = build_label_mapping(y_str)
    num_classes = len(str2idx)
    y_idx = np.vectorize(str2idx.get)(y_str)

    # Display class mappings for clarity at startup
    print("\nClass mapping (index -> label):")
    for i in range(num_classes):
        print(f"  {i}: {idx2str[i]}")
    print("Class mapping (label -> index):")
    for label in sorted(str2idx.keys()):
        print(f"  {label}: {str2idx[label]}")

    cfg = MultiModalConfig(num_classes=num_classes)
    # Apply CLI overrides to nested TOF config
    if args.tof_img_size and args.tof_img_size > 0:
        cfg.tof_cfg.img_size = int(args.tof_img_size)
    if args.tof_model_name:
        cfg.tof_cfg.model_name = args.tof_model_name
    # Handle both original flag and alias
    if args.no_freeze_tof_backbone or args.no_freeze_backbone:
        cfg.freeze_tof_backbone = False
    print(f"Model configuration: {cfg}")

    best_scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n======== Training Fold {fold+1}/{len(splits)} ========")
        # If user requested a specific num_workers, override auto heuristic
        if args.num_workers and args.num_workers > 0:
            os.environ["NUM_WORKERS_OVERRIDE"] = str(args.num_workers)
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
            optimizer_type=args.optimizer,
            scheduler_type=args.scheduler,
        )
        best_scores.append(best_f1)
        print(f"Fold {fold} final val_hier_f1: {best_f1:.4f}")
        flush()

    print("\n=== Cross-validated Hierarchical F1 ===")
    for i, s in enumerate(best_scores):
        print(f"Fold {i}: {s:.4f}")
    print(f"Mean F1: {np.mean(best_scores):.4f}")


if __name__ == "__main__":
    main()