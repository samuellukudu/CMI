"""PyTorch Dataset + simple sanity test for binary BFRB classification.

Relies on preprocessed artifacts created by `preprocessing.py` and saved under
`preprocessed/`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Sequence

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------

class BinaryIMUDataset(Dataset):
    """Loads IMU feature arrays and binary targets for BFRB vs non-BFRB."""

    def __init__(
        self,
        imu_array: np.ndarray,
        targets: Sequence[int] | np.ndarray,
    ) -> None:
        assert len(imu_array) == len(targets), "IMU array and targets length mismatch"
        self.imu_array = imu_array.astype(np.float32)  # (N, F)
        self.targets = np.asarray(targets, dtype=np.float32)  # (N,)

        # Reshape to (channels, timesteps) if already shaped (N, C, T)
        if self.imu_array.ndim == 2:
            # Assume flattened -> one timestep; add fake sequence length of 1
            self.imu_array = self.imu_array[:, :, None]
        elif self.imu_array.ndim == 3:
            pass  # expected shape
        else:
            raise ValueError("IMU array must be of shape (N, C) or (N, C, T)")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.imu_array[idx])  # (C, T)
        y = torch.tensor(self.targets[idx])  # scalar
        return x, y


# ---------------------------------------------------------------------------
#  Quick test utility
# ---------------------------------------------------------------------------

def _load_artifacts(root: str | Path = "preprocessed"):
    root = Path(root)
    imu = np.load(root / "train_imu.npy", mmap_mode="r")  # shape (N, F)
    cv = joblib.load(root / "cv_splits.joblib")
    y = cv["binary_targets"]  # shape (N,)
    return imu, y


def _sanity_check(batch_size: int = 32, device: str = "cpu") -> None:
    from imu_model import BinaryIMUCNN, BinaryIMUConfig

    print("Loading artifacts …")
    imu, targets = _load_artifacts()
    ds = BinaryIMUDataset(imu, targets)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    cfg = BinaryIMUConfig(in_channels=imu.shape[1])
    model = BinaryIMUCNN(cfg).to(device)

    # One forward & loss
    x, y = next(iter(dl))
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = model.loss(logits.squeeze(-1), y)
    print("Batch logits shape:", logits.shape)
    print("Loss:", loss.item())


if __name__ == "__main__":
    _sanity_check()