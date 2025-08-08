"""K-Fold training loop for masked sensor value imputation.

Loads preprocessed IMU, THM, TOF data + CV splits, trains MaskedImputer
on each fold, reports validation MSE losses.
"""
from __future__ import annotations

import argparse
import os
import random
import gc
from pathlib import Path
from typing import Tuple, Dict

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm, trange

# -------------------------------------------------------------
#  Reproducibility & memory helpers
# -------------------------------------------------------------

def seeding(seed: int) -> None:
    """Global deterministic behaviour for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False  # allows faster convs
        torch.backends.cudnn.benchmark = True
    print(f"Seeding done with seed={seed}")


def flush() -> None:
    """Release cached GPU memory to mitigate OOM between folds."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from masked_dataset import SensorMaskedDataset
from masked_model import MaskedImputer, MaskedImputerConfig

ROOT = Path(__file__).parent
PREP = ROOT / "preprocessed"


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE, handling zero values."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    nonzero_mask = y_true != 0
    if not np.any(nonzero_mask):
        return np.nan
    return np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100


def evaluate_reconstruction(model, val_loader, thm_scaler, tof_scaler, thm_orig_val, tof_orig_val, device):
    """Evaluate reconstruction quality on original scale using R2, MSE, MAE, MAPE."""
    model.eval()
    all_thm_pred = []
    all_tof_pred = []
    all_thm_true = []
    all_tof_true = []
    
    with torch.no_grad():
        for batch in val_loader:
            imu_batch = batch["imu"].to(device)
            thm_pred, tof_pred = model(imu_batch)
            
            # Store predictions and targets (scaled)
            all_thm_pred.append(thm_pred.cpu().numpy())
            all_tof_pred.append(tof_pred.cpu().numpy())
            all_thm_true.append(batch["thm_target"].numpy())
            all_tof_true.append(batch["tof_target"].numpy())
    
    # Concatenate all batches
    thm_pred_scaled = np.concatenate(all_thm_pred, axis=0)
    tof_pred_scaled = np.concatenate(all_tof_pred, axis=0)
    thm_true_scaled = np.concatenate(all_thm_true, axis=0)
    tof_true_scaled = np.concatenate(all_tof_true, axis=0)
    
    # Inverse transform to original scale
    thm_pred_orig = thm_scaler.inverse_transform(thm_pred_scaled)
    tof_pred_orig = tof_scaler.inverse_transform(tof_pred_scaled)
    thm_true_orig = thm_scaler.inverse_transform(thm_true_scaled)
    tof_true_orig = tof_scaler.inverse_transform(tof_true_scaled)
    
    # Calculate metrics for THM
    thm_r2 = r2_score(thm_true_orig.flatten(), thm_pred_orig.flatten())
    thm_mse = mean_squared_error(thm_true_orig.flatten(), thm_pred_orig.flatten())
    thm_mae = mean_absolute_error(thm_true_orig.flatten(), thm_pred_orig.flatten())
    thm_mape = mean_absolute_percentage_error(thm_true_orig.flatten(), thm_pred_orig.flatten())
    
    # Calculate metrics for TOF
    tof_r2 = r2_score(tof_true_orig.flatten(), tof_pred_orig.flatten())
    tof_mse = mean_squared_error(tof_true_orig.flatten(), tof_pred_orig.flatten())
    tof_mae = mean_absolute_error(tof_true_orig.flatten(), tof_pred_orig.flatten())
    tof_mape = mean_absolute_percentage_error(tof_true_orig.flatten(), tof_pred_orig.flatten())
    
    return {
        "thm_r2": thm_r2,
        "thm_mse": thm_mse,
        "thm_mae": thm_mae,
        "thm_mape": thm_mape,
        "tof_r2": tof_r2,
        "tof_mse": tof_mse,
        "tof_mae": tof_mae,
        "tof_mape": tof_mape,
    }


def load_artifacts() -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, np.ndarray, np.ndarray]:
    """Load preprocessed arrays, CV splits, and original unscaled data."""
    imu = np.load(PREP / "train_imu.npy")  # (N, F)
    thm = np.load(PREP / "train_thm.npy")  # (N, F_thm)
    tof = np.load(PREP / "train_tof.npy")  # (N, F_tof)
    
    # Load original unscaled data for evaluation
    thm_orig = np.load(PREP / "train_thm_unscaled.npy")
    tof_orig = np.load(PREP / "train_tof_unscaled.npy")
    
    cv = joblib.load(PREP / "cv_splits.joblib")
    splits = cv["binary_splits"]  # Reuse same splits as binary classification
    return imu, thm, tof, splits, thm_orig, tof_orig


def train_fold(
    fold: int, 
    train_idx: np.ndarray, 
    val_idx: np.ndarray, 
    imu: np.ndarray, 
    thm: np.ndarray,
    tof: np.ndarray,
    thm_orig: np.ndarray,
    tof_orig: np.ndarray,
    cfg: MaskedImputerConfig, 
    epochs: int, 
    batch_size: int, 
    lr: float, 
    device: str,
    mask_ratio: float = 0.65,
) -> Dict[str, float]:
    """Train and evaluate a single fold."""
    # Load scalers for inverse transformation
    thm_scaler = joblib.load(PREP / "thm_scaler.joblib")
    tof_scaler = joblib.load(PREP / "tof_scaler.joblib")
    
    # Datasets
    train_ds = SensorMaskedDataset(imu[train_idx], thm[train_idx], tof[train_idx], mask_ratio=mask_ratio)
    val_ds = SensorMaskedDataset(imu[val_idx], thm[val_idx], tof[val_idx], mask_ratio=mask_ratio)
    
    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = MaskedImputer(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine annealing scheduler for smooth LR decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=epochs
    )
    
    best_val_loss = float("inf")
    best_metrics = {}
    
    for epoch in trange(epochs, desc=f"Fold {fold}"):
        # ---------- Training ----------
        model.train()
        running_loss = 0.0
        running_thm_loss = 0.0
        running_tof_loss = 0.0
        n_samples = 0

        for batch in tqdm(train_loader, desc="Train", leave=False):
            # Move data to device
            imu_batch = batch["imu"].to(device)
            thm_target = batch["thm_target"].to(device)
            tof_target = batch["tof_target"].to(device)
            thm_mask = batch["thm_mask"].to(device)
            tof_mask = batch["tof_mask"].to(device)
            
            # Forward pass
            opt.zero_grad()
            thm_pred, tof_pred = model(imu_batch)
            
            # Calculate loss (only on masked positions)
            loss, thm_loss_item, tof_loss_item = model.reconstruction_loss(
                (thm_pred, tof_pred),
                (thm_target, tof_target),
                (thm_mask, tof_mask)
            )
            
            # Backward pass
            loss.backward()
            opt.step()
            
            # Track losses
            batch_size = imu_batch.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size
            
            # Track component losses (already computed in reconstruction_loss)
            running_thm_loss += thm_loss_item * batch_size
            running_tof_loss += tof_loss_item * batch_size

        # Compute epoch metrics
        train_loss = running_loss / n_samples
        train_thm_loss = running_thm_loss / n_samples
        train_tof_loss = running_tof_loss / n_samples

        # ---------- Validation ----------
        model.eval()
        val_running_loss = 0.0
        val_running_thm_loss = 0.0
        val_running_tof_loss = 0.0
        val_n_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                # Move data to device
                imu_batch = batch["imu"].to(device)
                thm_target = batch["thm_target"].to(device)
                tof_target = batch["tof_target"].to(device)
                thm_mask = batch["thm_mask"].to(device)
                tof_mask = batch["tof_mask"].to(device)
                
                # Forward pass
                thm_pred, tof_pred = model(imu_batch)
                
                # Calculate loss
                loss, thm_loss_item, tof_loss_item = model.reconstruction_loss(
                    (thm_pred, tof_pred),
                    (thm_target, tof_target),
                    (thm_mask, tof_mask)
                )
                
                # Track losses
                batch_size = imu_batch.size(0)
                val_running_loss += loss.item() * batch_size
                val_n_samples += batch_size
                
                # Track component losses (already computed in reconstruction_loss)
                val_running_thm_loss += thm_loss_item * batch_size
                val_running_tof_loss += tof_loss_item * batch_size

        # Compute validation metrics
        val_loss = val_running_loss / val_n_samples
        val_thm_loss = val_running_thm_loss / val_n_samples
        val_tof_loss = val_running_tof_loss / val_n_samples
        
        # Update learning rate
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                "val_loss": val_loss,
                "val_thm_loss": val_thm_loss,
                "val_tof_loss": val_tof_loss,
            }
            # Save model checkpoint
            os.makedirs(ROOT / "checkpoints", exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "metrics": best_metrics,
            }, ROOT / f"checkpoints/masked_imputer_fold{fold}.pt")

        # Print metrics
        tqdm.write(
            f"Fold {fold} Epoch {epoch+1}/{epochs} - "
            f"loss: {train_loss:.4f} (thm: {train_thm_loss:.4f}, tof: {train_tof_loss:.4f}) - "
            f"val_loss: {val_loss:.4f} (thm: {val_thm_loss:.4f}, tof: {val_tof_loss:.4f})"
        )

    # Evaluate reconstruction quality on original scale
    print(f"\nEvaluating reconstruction quality for fold {fold}...")
    reconstruction_metrics = evaluate_reconstruction(
        model, val_loader, thm_scaler, tof_scaler, 
        thm_orig[val_idx], tof_orig[val_idx], device
    )
    
    # Print reconstruction metrics
    print(f"Fold {fold} Reconstruction Metrics:")
    print(f"  THM - R2: {reconstruction_metrics['thm_r2']:.4f}, MSE: {reconstruction_metrics['thm_mse']:.4f}, "
          f"MAE: {reconstruction_metrics['thm_mae']:.4f}, MAPE: {reconstruction_metrics['thm_mape']:.2f}%")
    print(f"  TOF - R2: {reconstruction_metrics['tof_r2']:.4f}, MSE: {reconstruction_metrics['tof_mse']:.4f}, "
          f"MAE: {reconstruction_metrics['tof_mae']:.4f}, MAPE: {reconstruction_metrics['tof_mape']:.2f}%")
    
    # Add reconstruction metrics to best_metrics
    best_metrics.update(reconstruction_metrics)
    
    return best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    # Device detection: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        default_device = "cuda"
    elif torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mask_ratio", type=float, default=0.65, 
                        help="Mask ratio for sensor data (0.5-0.8)")
    parser.add_argument("--loss_type", choices=["mae", "huber"], default="mae",
                        help="Loss function type")
    parser.add_argument("--huber_beta", type=float, default=1.0,
                        help="Beta parameter for Huber loss")
    args = parser.parse_args()
    
    # Validate mask_ratio
    if not (0.15 <= args.mask_ratio <= 0.8):
        raise ValueError(f"mask_ratio must be between 0.15 and 0.8, got {args.mask_ratio}")

    # Set random seed
    seeding(args.seed)

    # Load data
    print("Loading preprocessed data...")
    imu, thm, tof, splits, thm_orig, tof_orig = load_artifacts()
    
    # Create model config
    cfg = MaskedImputerConfig(
        in_channels_imu=imu.shape[1],
        thm_out_dim=thm.shape[1],
        tof_out_dim=tof.shape[1],
        loss_type=args.loss_type,
        huber_beta=args.huber_beta,
    )
    print(f"Model configuration: {cfg}")

    # Training loop
    results = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n======== Training Fold {fold+1}/{len(splits)} ========")
        metrics = train_fold(
            fold, train_idx, val_idx, imu, thm, tof, thm_orig, tof_orig, cfg, 
            args.epochs, args.batch_size, args.lr, args.device,
            mask_ratio=args.mask_ratio,
        )
        results.append(metrics)
        # Flush GPU memory between folds
        flush()

    # Print cross-validation results
    print("\n===== Cross-validation Results =====")
    all_val_losses = [r["val_loss"] for r in results]
    all_val_thm_losses = [r["val_thm_loss"] for r in results]
    all_val_tof_losses = [r["val_tof_loss"] for r in results]
    
    # Reconstruction metrics
    all_thm_r2 = [r["thm_r2"] for r in results]
    all_thm_mse = [r["thm_mse"] for r in results]
    all_thm_mae = [r["thm_mae"] for r in results]
    all_thm_mape = [r["thm_mape"] for r in results]
    
    all_tof_r2 = [r["tof_r2"] for r in results]
    all_tof_mse = [r["tof_mse"] for r in results]
    all_tof_mae = [r["tof_mae"] for r in results]
    all_tof_mape = [r["tof_mape"] for r in results]
    
    for i, metrics in enumerate(results):
        print(f"Fold {i} - Val Loss: {metrics['val_loss']:.4f} "
              f"(THM: {metrics['val_thm_loss']:.4f}, TOF: {metrics['val_tof_loss']:.4f})")
        print(f"        Reconstruction - THM R2: {metrics['thm_r2']:.4f}, TOF R2: {metrics['tof_r2']:.4f}")
    
    print(f"\nAverage - Val Loss: {np.mean(all_val_losses):.4f} "
          f"(THM: {np.mean(all_val_thm_losses):.4f}, TOF: {np.mean(all_val_tof_losses):.4f})")
    
    print("\n===== Reconstruction Quality (Original Scale) =====")
    print(f"THM - R2: {np.mean(all_thm_r2):.4f}±{np.std(all_thm_r2):.4f}, "
          f"MSE: {np.mean(all_thm_mse):.4f}±{np.std(all_thm_mse):.4f}, "
          f"MAE: {np.mean(all_thm_mae):.4f}±{np.std(all_thm_mae):.4f}, "
          f"MAPE: {np.mean(all_thm_mape):.2f}±{np.std(all_thm_mape):.2f}%")
    print(f"TOF - R2: {np.mean(all_tof_r2):.4f}±{np.std(all_tof_r2):.4f}, "
          f"MSE: {np.mean(all_tof_mse):.4f}±{np.std(all_tof_mse):.4f}, "
          f"MAE: {np.mean(all_tof_mae):.4f}±{np.std(all_tof_mae):.4f}, "
          f"MAPE: {np.mean(all_tof_mape):.2f}±{np.std(all_tof_mape):.2f}%")


if __name__ == "__main__":
    main()