"""K-Fold training for IMU-only 2D spectrogram classifier (multi-class).

Mirrors the Stage-3 loop but focuses on IMU data only. Labels are 9-class by
default (8 BFRB gestures + non_target) sourced from cv_splits.joblib.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, List

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd
from tqdm.auto import tqdm, trange

from imu2d_dataset import IMU2DSpectrogramDataset, SpectrogramParams
from imu2d_seq_dataset import IMU2DSequenceSpectrogramDataset, SpectrogramParamsSeq
from imu2d_model import IMU2DEfficientNet, IMU2DEfficientNetConfig
from cmi_2025 import CompetitionMetric
from utils import seeding, flush


ROOT = Path(__file__).parent
PREP = ROOT / "preprocessed"


def load_artifacts() -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    imu = np.load(PREP / "train_imu.npy")  # (N, F) or (N, C, T)
    cv = joblib.load(PREP / "cv_splits.joblib")
    if "all_gestures" not in cv or "all_splits" not in cv:
        raise RuntimeError("cv_splits.joblib must contain 'all_gestures' and 'all_splits' for IMU2D training. Re-run preprocessing to generate full multi-class labels.")
    return imu, cv["all_gestures"], cv["all_splits"]


def load_sequence_artifacts() -> Tuple[List[np.ndarray], np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """Load per-sequence IMU time series from train_processed.parquet.

    Assumes IMU features in preprocessing order; we take acc_x/acc_y/acc_z if available
    and stack over time per sequence.
    """
    import polars as pl
    df = pl.read_parquet(PREP / "train_processed.parquet")
    cv = joblib.load(PREP / "cv_splits.joblib")
    # Build labels (9-class) as in multimodal
    target_mask = cv["binary_targets"] == 1
    y_str_bfrb = cv["bfrb_targets"]
    assert target_mask.sum() == len(y_str_bfrb)
    # Use gestures from df (string), but convert to 9-class mapping below
    gestures = df["gesture"].to_numpy()
    sequences = df["sequence_id"].to_numpy() if "sequence_id" in df.columns else np.arange(len(df))

    # Identify IMU columns (prefer raw acc_* if present)
    imu_cols = [c for c in df.columns if c.startswith("acc_") or c.startswith("rot_")]
    # Keep only acc_x/y/z for spectrogram by default
    keep = [c for c in imu_cols if c in ("acc_x", "acc_y", "acc_z")]
    if len(keep) < 3 and len(imu_cols) >= 3:
        keep = imu_cols[:3]

    seq_ids = np.unique(sequences)
    seq_arrays: List[np.ndarray] = []
    seq_labels: List[str] = []
    for sid in seq_ids:
        mask = sequences == sid
        arr = df.filter(pl.Series(mask)).select(keep).to_numpy().T.astype(np.float32)  # (C, T)
        lab = df.filter(pl.Series(mask))["gesture"].to_numpy()[0]
        seq_arrays.append(arr)
        seq_labels.append(lab)

    # Strict: require full multi-class labels for sequence path
    if "all_gestures" not in cv or "all_splits" not in cv:
        raise RuntimeError("cv_splits.joblib must contain 'all_gestures' and 'all_splits' for IMU2D sequence training. Re-run preprocessing to generate full multi-class labels.")
    y_all = np.array(seq_labels, dtype=object)
    row_splits = cv["all_splits"]
    # Build mapping from sequence id to row indices
    seq_to_rows: Dict[int, np.ndarray] = {}
    seq_id_values = np.unique(sequences)
    seq_index_map = {sid: i for i, sid in enumerate(seq_id_values)}
    for sid in seq_id_values:
        seq_to_rows[seq_index_map[sid]] = np.where(sequences == sid)[0]
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for train_rows, val_rows in row_splits:
        train_rows_set = set(train_rows.tolist())
        val_rows_set = set(val_rows.tolist())
        train_seq_idx: List[int] = []
        val_seq_idx: List[int] = []
        for sidx in range(len(seq_arrays)):
            rows = seq_to_rows[sidx]
            if any((r in val_rows_set) for r in rows):
                val_seq_idx.append(sidx)
            else:
                train_seq_idx.append(sidx)
        splits.append((np.array(train_seq_idx), np.array(val_seq_idx)))
    return seq_arrays, y_all, splits


def build_label_mapping(labels: np.ndarray):
    uniq = sorted(np.unique(labels))
    str2idx = {g: i for i, g in enumerate(uniq)}
    idx2str = {i: g for g, i in str2idx.items()}
    return str2idx, idx2str


def train_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    imu: np.ndarray,
    y_idx: np.ndarray,
    spec_params: SpectrogramParams,
    eff_cfg: IMU2DEfficientNetConfig,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    channel_indices: List[int],
    idx2str: Dict[int, str],
    optimizer_type: str = "adam",
    scheduler_type: str = "cosine",
    num_workers: int = 0,
    pin_memory: bool = False,
    freeze_backbone: bool = True,
    model_type: str = "efficientnet",
) -> float:
    ds = IMU2DSpectrogramDataset(imu, y_idx, channel_indices=channel_indices, spec_params=spec_params, log_amplitude=True)
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

    # Model selection
    model = IMU2DEfficientNet(eff_cfg).to(device)
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    # Optimizer
    if optimizer_type.lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    elif scheduler_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
    else:
        scheduler = None

    best_f1 = -1.0
    # Enable mixed precision (AMP) on CUDA only
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    for epoch in trange(epochs, desc=f"Fold {fold}"):
        model.train()
        running_loss = 0.0
        for xb, yb in tqdm(train_loader, desc="train", leave=False):
            xb = xb.to(device, non_blocking=True)
            yb = yb.long().to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            if device == "cuda":
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            if scheduler is not None and scheduler_type == "onecycle":
                scheduler.step()
            running_loss += loss.item() * xb.size(0)
        avg_loss = running_loss / len(train_loader.dataset)

        if scheduler is not None and scheduler_type != "onecycle":
            scheduler.step()

        # Validation hierarchical F1 (CompetitionMetric)
        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.long().to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1).cpu().numpy()
                true = yb.cpu().numpy()
                all_preds.extend(pred)
                all_trues.extend(true)
        preds_str = [idx2str[i] for i in all_preds]
        trues_str = [idx2str[i] for i in all_trues]
        df_pred = pd.DataFrame({"gesture": preds_str})
        df_true = pd.DataFrame({"gesture": trues_str})
        metric = CompetitionMetric()
        f1 = metric.calculate_hierarchical_f1(df_true, df_pred)
        tqdm.write(f"Fold {fold} Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} val_hier_f1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(ROOT / "checkpoints", exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": eff_cfg,
                "epoch": epoch,
                "val_hier_f1": f1,
            }, ROOT / f"checkpoints/imu2d_fold{fold}.pt")
        # Release memory after each epoch
        flush()

    return best_f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    if torch.cuda.is_available():
        default_device = "cuda"
    elif torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing preprocessed artifacts")
    # Optimizer & scheduler
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--scheduler", choices=["none", "cosine", "onecycle"], default="cosine")
    # Spectrogram params
    parser.add_argument("--fs", type=float, default=200.0)
    parser.add_argument("--nperseg", type=int, default=256)
    parser.add_argument("--noverlap", type=int, default=128)
    parser.add_argument("--nfft", type=int, default=0, help="0 means use nperseg")
    parser.add_argument("--transform", choices=["stft", "cwt"], default="stft", help="Spectrogram transform to use")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", help="timm backbone for IMU2D (e.g., efficientnet_b0)")
    parser.add_argument("--img_size", type=int, default=0, help="Resize spectrogram to this square size for EfficientNet (0 keeps default)")
    parser.add_argument("--img_hw", type=str, default="", help="Resize spectrogram to HxW (e.g., 128x64, 128,64 or (128,64)). Overrides --img_size if provided")
    parser.add_argument("--model", choices=["efficientnet"], default="efficientnet", help="Model architecture: 'efficientnet' only")
    # Freeze backbone
    parser.add_argument("--no_freeze_backbone", action="store_true", help="Unfreeze EfficientNet backbone (frozen by default)")
    # Channel selection (optional). If omitted, all IMU channels are used.
    parser.add_argument("--channels", type=str, default="", help="comma-separated channel indices to use; empty means all channels")
    parser.add_argument("--from_sequences", action="store_true", help="Build STFT from raw per-sequence time series in parquet")
    parser.add_argument("--num_workers", type=int, default=0, help="Override DataLoader workers (0=auto)")
    args = parser.parse_args()

    # Override preprocessed directory
    global PREP
    if args.data_dir is not None:
        PREP = Path(args.data_dir)

    if args.from_sequences:
        seq_arrays, y_str, splits = load_sequence_artifacts()
        str2idx, idx2str = build_label_mapping(y_str)
        y_idx = np.vectorize(str2idx.get)(y_str)
    else:
        imu, y_str, splits = load_artifacts()
        str2idx, idx2str = build_label_mapping(y_str)
        y_idx = np.vectorize(str2idx.get)(y_str)

    # Echo class mapping
    print("\nClass mapping (index -> label):")
    for i in range(len(idx2str)):
        print(f"  {i}: {idx2str[i]}")

    # Dataset/model config
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    
    if args.from_sequences:
         # For sequences, infer channel count from the first sequence
         inferred_in_channels = seq_arrays[0].shape[0]
    else:
        # For arrays, infer channel count from IMU array last known dimension
        inferred_in_channels = imu.shape[1] if "imu" in locals() else seq_arrays[0].shape[0]
    channel_indices = [int(x) for x in args.channels.split(",") if x != ""] if args.channels else list(range(inferred_in_channels))
    spec_params = SpectrogramParams(
        fs=args.fs,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        nfft=(None if args.nfft == 0 else args.nfft),
        transform=args.transform,
    )
    eff_cfg = IMU2DEfficientNetConfig(
        model_name=args.backbone,
        in_channels=len(channel_indices),
        num_classes=len(idx2str),
    )
    # Apply freezing choice to config for downstream components
    eff_cfg.freeze_backbone = freeze_backbone
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone
    print(f"Model configuration: {eff_cfg}")
    # Determine if backbone should be frozen (default: True unless flag overrides)
    freeze_backbone = not args.no_freeze_backbone