"""PyTorch Dataset for masked sensor reconstruction (Stage-1 imputation).

This dataset prepares inputs/targets for a masked-modeling task where a large
portion (50-80 %) of THM and TOF sensor values are randomly hidden. The model
sees IMU data + the partially-masked THM/TOF tensors and must reconstruct the
missing values.

The class works purely on memory-mapped numpy arrays produced by
`preprocessing.py` and stored under the default ``preprocessed/`` folder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Sequence, Union, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------


class SensorMaskedDataset(Dataset):
    """Dataset that applies random masking to THM and TOF sensors.

    Parameters
    ----------
    imu_array, thm_array, tof_array
        Memory-mapped or in-memory numpy arrays with **matching length** on the
        first dimension (N, …). Shapes are `(N, C_imu)`, `(N, C_thm)` and
        `(N, C_tof)` respectively.
    thm_observed, tof_observed : Optional[np.ndarray]
        Binary masks indicating observed data (1) vs missing (0) before imputation.
        If provided, these are combined with random masking for training.
    mask_ratio : float | Tuple[float, float]
        Either a fixed mask ratio (0.0-1.0) or a *(low, high)* range.  For a
        tuple, a new ratio is sampled *per-instance* from
        ``Uniform(low, high)``.
    mask_value : float
        Value used to replace the masked elements. 0.0 is a safe default.
    """

    def __init__(
        self,
        imu_array: np.ndarray,
        thm_array: np.ndarray,
        tof_array: np.ndarray,
        thm_observed: Optional[np.ndarray] = None,
        tof_observed: Optional[np.ndarray] = None,
        mask_ratio: Union[float, Tuple[float, float]] = (0.5, 0.8),
        mask_value: float = 0.0,
    ) -> None:
        assert len(imu_array) == len(thm_array) == len(tof_array), "Arrays length mismatch"

        # Store as float32 to save memory + ensure torch compatibility
        self.imu_array = imu_array.astype(np.float32)
        self.thm_array = thm_array.astype(np.float32)
        self.tof_array = tof_array.astype(np.float32)
        
        # Store observed masks if provided
        self.thm_observed = thm_observed.astype(np.float32) if thm_observed is not None else None
        self.tof_observed = tof_observed.astype(np.float32) if tof_observed is not None else None
        
        if self.thm_observed is not None:
            assert len(self.thm_observed) == len(thm_array), "THM observed mask length mismatch"
        if self.tof_observed is not None:
            assert len(self.tof_observed) == len(tof_array), "TOF observed mask length mismatch"

        if isinstance(mask_ratio, tuple):
            lo, hi = mask_ratio
            assert 0.0 <= lo < hi <= 1.0, "mask_ratio tuple must be within [0,1] and lo < hi"
        else:
            assert 0.0 <= mask_ratio <= 1.0, "mask_ratio must be within [0,1]"
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value

    # ---------------------------------------------------------------------
    #  Dataset protocol
    # ---------------------------------------------------------------------

    def __len__(self) -> int:  # noqa: D401 (simple returns preferred)
        return len(self.imu_array)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Fetch raw values first (keep copies for targets)
        imu = self.imu_array[idx]  # (C_imu,)
        thm = self.thm_array[idx]  # (C_thm,)
        tof = self.tof_array[idx]  # (C_tof,)

        # Sample ratio per instance if a range was provided
        ratio = (
            np.random.uniform(*self.mask_ratio)
            if isinstance(self.mask_ratio, tuple)
            else self.mask_ratio
        )

        # -----------------------------------------------------------------
        #  Build loss masks (random on observed) and input masks (orig missing OR random)
        # -----------------------------------------------------------------
        # THM
        if self.thm_observed is not None:
            thm_obs_mask = self.thm_observed[idx]  # 1 = observed, 0 = originally missing
            thm_random_mask = (np.random.rand(*thm.shape) < ratio) & (thm_obs_mask == 1)
            thm_input_mask = (thm_obs_mask == 0) | thm_random_mask
            thm_loss_mask = thm_random_mask
        else:
            thm_random_mask = np.random.rand(*thm.shape) < ratio
            thm_input_mask = thm_random_mask
            thm_loss_mask = thm_random_mask
        
        # TOF
        if self.tof_observed is not None:
            tof_obs_mask = self.tof_observed[idx]  # 1 = observed, 0 = originally missing
            tof_random_mask = (np.random.rand(*tof.shape) < ratio) & (tof_obs_mask == 1)
            tof_input_mask = (tof_obs_mask == 0) | tof_random_mask
            tof_loss_mask = tof_random_mask
        else:
            tof_random_mask = np.random.rand(*tof.shape) < ratio
            tof_input_mask = tof_random_mask
            tof_loss_mask = tof_random_mask

        # Create masked copies for model input (mask both originally missing and random)
        thm_in = thm.copy()
        thm_in[thm_input_mask] = self.mask_value
        tof_in = tof.copy()
        tof_in[tof_input_mask] = self.mask_value

        return {
            "imu": torch.from_numpy(imu),
            "thm_input": torch.from_numpy(thm_in),
            "tof_input": torch.from_numpy(tof_in),
            "thm_target": torch.from_numpy(thm),
            "tof_target": torch.from_numpy(tof),
            # Loss masks: 1 at randomly masked observed positions only
            "thm_mask": torch.from_numpy(thm_loss_mask.astype(np.float32)),
            "tof_mask": torch.from_numpy(tof_loss_mask.astype(np.float32)),
        }

    # ---------------------------------------------------------------------
    #  Convenience factory
    # ---------------------------------------------------------------------

    @classmethod
    def from_preprocessed(
        cls,
        root: str | Path = "preprocessed",
        mask_ratio: Union[float, Tuple[float, float]] = (0.5, 0.8),
        mask_value: float = 0.0,
        use_observed_masks: bool = True,
    ) -> "SensorMaskedDataset":
        """Instantiate the dataset directly from the *preprocessed* folder."""
        root = Path(root)
        imu = np.load(root / "train_imu.npy", mmap_mode="r")
        thm = np.load(root / "train_thm.npy", mmap_mode="r")
        tof = np.load(root / "train_tof.npy", mmap_mode="r")
        
        # Load observed masks if requested and available
        thm_observed = None
        tof_observed = None
        if use_observed_masks:
            try:
                thm_observed = np.load(root / "train_thm_observed.npy", mmap_mode="r")
                tof_observed = np.load(root / "train_tof_observed.npy", mmap_mode="r")
            except FileNotFoundError:
                print("Warning: Observed masks not found, using pure random masking")
        
        return cls(imu, thm, tof, thm_observed, tof_observed, 
                  mask_ratio=mask_ratio, mask_value=mask_value)


# ---------------------------------------------------------------------------
#  Quick sanity test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = SensorMaskedDataset.from_preprocessed()
    sample = ds[0]
    for k, v in sample.items():
        print(k, v.shape, v.dtype, v.min().item(), v.max().item())