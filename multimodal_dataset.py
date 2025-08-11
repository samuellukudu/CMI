from __future__ import annotations

"""PyTorch Dataset for unified multi-sensor gesture classification (Stage-3).

Each item returns a tuple (imu, thm, tof, label) where
    imu  : torch.Tensor shaped (C_imu, T)
    thm  : torch.Tensor shaped (T, F_thm)
    tof  : torch.Tensor shaped (5, 8, 8)
    label: torch.Tensor scalar (int or float)

The class expects preprocessed numpy arrays produced by ``preprocessing.py``
and stored under the default ``preprocessed/`` directory. Arrays must have
**matching length** N on the first dimension.

Example
-------
>>> ds = MultiSensorDataset(imu, thm, tof, labels)
>>> imu_x, thm_x, tof_x, y = ds[0]
>>> print(imu_x.shape, thm_x.shape, tof_x.shape, y)

This dataset performs *no* on-the-fly augmentation or scaling—apply those in a
``DataLoader`` collate_fn or transform wrapper if required.
"""

from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------


class MultiSensorDataset(Dataset):
    """Unified dataset yielding IMU, THM, TOF tensors and gesture labels."""

    def __init__(
        self,
        imu_array: np.ndarray,
        thm_array: np.ndarray,
        tof_array: np.ndarray,
        labels: Sequence[Union[int, float, str]] | np.ndarray,
        bfrb_only: bool = False,
        non_target_label: Union[int, float, str] = "non_target",
    ) -> None:
        # ---------------- Optional filtering ----------------
        labels_arr = np.asarray(labels)
        if bfrb_only:
            keep_mask = labels_arr != non_target_label
            imu_array = imu_array[keep_mask]
            thm_array = thm_array[keep_mask]
            tof_array = tof_array[keep_mask]
            labels_arr = labels_arr[keep_mask]

        # ---------------- Sanity checks ----------------
        if not (len(imu_array) == len(thm_array) == len(tof_array) == len(labels_arr)):
            raise ValueError("All input arrays must share the same first-dim length")

        # Cast to float32 for torch friendliness
        self.imu_array = imu_array.astype(np.float32)
        self.thm_array = thm_array.astype(np.float32)
        self.tof_array = tof_array.astype(np.float32)
        self.labels = labels_arr.astype(np.float32) if labels_arr.dtype.kind == "f" else labels_arr

        # ------------- Shape normalisation -------------
        # IMU: expected (N, C, T). Accept flattened (N, C) ⇒ add fake T=1.
        if self.imu_array.ndim == 2:  # (N, C)
            self.imu_array = self.imu_array[:, :, None]  # -> (N, C, 1)
        elif self.imu_array.ndim != 3:
            raise ValueError("IMU array must have shape (N, C) or (N, C, T)")

        # THM: expected (N, T, F). Accept flattened (N, F) ⇒ add fake T=1.
        if self.thm_array.ndim == 2:  # (N, F)
            self.thm_array = self.thm_array[:, None, :]  # -> (N, 1, F)
        elif self.thm_array.ndim != 3:
            raise ValueError("THM array must have shape (N, T, F) or (N, F)")

        # TOF: expected (N, 5, 8, 8). Accept flattened (N, 320) or (N, 5*8*8)
        if self.tof_array.ndim == 2:  # (N, 320)
            if self.tof_array.shape[1] != 5 * 8 * 8:
                raise ValueError(
                    "Flattened TOF must have 320 = 5×8×8 features per instance"
                )
            self.tof_array = self.tof_array.reshape(-1, 5, 8, 8)
        elif self.tof_array.ndim != 4:
            raise ValueError("TOF array must be (N, 5, 8, 8) or flattened (N, 320)")

    # ---------------------------------------------------------------------
    #  Dataset protocol
    # ---------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        imu = torch.from_numpy(self.imu_array[idx])      # (C, T)
        thm = torch.from_numpy(self.thm_array[idx])      # (T, F)
        tof = torch.from_numpy(self.tof_array[idx])      # (5, 8, 8)
        label = torch.tensor(self.labels[idx])           # scalar
        return imu, thm, tof, label