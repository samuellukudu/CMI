from __future__ import annotations

"""Dataset that converts 1D IMU sequences to 2D spectrograms (STFT by default).

It consumes IMU arrays shaped (N, C, T) or (N, C) and labels (str or int), and
produces tensors (C_sel, F, TT) suitable for 2D CNNs.

Spectrogram parameters are configurable; SciPy is used for STFT.
"""

from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional, Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import stft
import pywt


@dataclass
class SpectrogramParams:
    fs: float = 200.0           # sampling rate (Hz)
    nperseg: int = 256          # window length (STFT)
    noverlap: int = 128         # hop = nperseg - noverlap (STFT)
    nfft: Optional[int] = None  # use nperseg if None (STFT)
    transform: Literal["stft", "cwt"] = "stft"


class IMU2DSpectrogramDataset(Dataset):
    def __init__(
        self,
        imu_array: np.ndarray,  # (N, C) or (N, C, T)
        labels: Sequence,       # str/int labels
        channel_indices: Optional[List[int]] = None,
        spec_params: SpectrogramParams = SpectrogramParams(),
        log_amplitude: bool = True,
        eps: float = 1e-6,
    ) -> None:
        assert imu_array.ndim in (2, 3), "IMU array must be (N, C) or (N, C, T)"
        if imu_array.ndim == 2:
            imu_array = imu_array[:, :, None]
        self.imu = imu_array.astype(np.float32)
        self.labels = np.asarray(labels)
        assert len(self.imu) == len(self.labels)

        # Default: use all IMU channels
        self.channel_indices = channel_indices if channel_indices is not None else list(range(self.imu.shape[1]))

        # ----------------------------
        # Sanity checks
        # ----------------------------
        assert len(self.channel_indices) > 0, "channel_indices cannot be empty"
        max_ch = self.imu.shape[1] - 1
        assert all(0 <= ch <= max_ch for ch in self.channel_indices), "channel_indices out of range"
        # Spectrogram parameter validation
        assert spec_params.fs > 0, "Sampling rate fs must be positive"
        assert spec_params.nperseg > 0, "nperseg must be > 0"
        assert 0 <= spec_params.noverlap < spec_params.nperseg, "noverlap must satisfy 0 <= noverlap < nperseg"
        if spec_params.nfft is not None:
            assert spec_params.nfft >= spec_params.nperseg, "nfft must be >= nperseg"

        self.spec_params = spec_params
        self.log_amplitude = log_amplitude
        self.eps = eps

        # Precompute spectrogram grid sizes using a short probe
        if self.spec_params.transform == "stft":
            x0 = self.imu[0, self.channel_indices[0]].astype(np.float32)
            local_nperseg = min(self.spec_params.nperseg, x0.shape[0])
            local_noverlap = min(self.spec_params.noverlap, max(0, local_nperseg - 1))
            _, _, Z = stft(
                x0,
                fs=self.spec_params.fs,
                nperseg=local_nperseg,
                noverlap=local_noverlap,
                nfft=self.spec_params.nfft or local_nperseg,
                boundary=None,
            )
            self.freq_bins = Z.shape[0]
            self.time_bins = Z.shape[1]
        else:
            widths = np.geomspace(1.5, 64.0, 64).astype(np.float32)
            self.freq_bins = len(widths)
            self.time_bins = self.imu.shape[-1]

    def __len__(self) -> int:
        return len(self.labels)

    def _spectrogram(self, x_1d: np.ndarray) -> np.ndarray:
        if self.spec_params.transform == "stft":
            local_nperseg = min(self.spec_params.nperseg, x_1d.shape[0])
            local_noverlap = min(self.spec_params.noverlap, max(0, local_nperseg - 1))
            f, t, Z = stft(
                x_1d.astype(np.float32),
                fs=self.spec_params.fs,
                nperseg=local_nperseg,
                noverlap=local_noverlap,
                nfft=self.spec_params.nfft or local_nperseg,
                boundary=None,
            )
            mag = np.abs(Z).astype(np.float32)
        else:
            widths = np.geomspace(1.5, 64.0, 64).astype(np.float32)
            coeffs, _ = pywt.cwt(x_1d.astype(np.float32), widths, 'morl', sampling_period=1.0 / self.spec_params.fs)
            mag = np.abs(coeffs).astype(np.float32)
        if self.log_amplitude:
            mag = np.log(mag + self.eps)
        return mag

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.imu[idx]  # (C_all, T)
        spec_list = []
        for ch in self.channel_indices:
            spec = self._spectrogram(x[ch])  # (F, TT)
            spec_list.append(spec)
        spec = np.stack(spec_list, axis=0)  # (C_sel, F, TT)
        y = self.labels[idx]
        return torch.from_numpy(spec), torch.tensor(y)


if __name__ == "__main__":
    # Quick sanity test for IMU2DSpectrogramDataset
    N, C, T = 8, 6, 1024
    rng = np.random.RandomState(0)
    imu = rng.randn(N, C, T).astype(np.float32)
    labels = np.arange(N)

    ds = IMU2DSpectrogramDataset(imu, labels)
    spec, lbl = ds[0]
    print("Spec shape:", spec.shape)
    print("Label:", lbl)
    print("Freq bins / Time bins:", ds.freq_bins, ds.time_bins)