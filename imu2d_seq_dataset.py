from __future__ import annotations

"""Sequence-based IMU 2D spectrogram dataset.

Consumes per-sequence time series (list of arrays shaped (C, T)) and labels,
producing STFT spectrogram tensors (C_sel, F, TT).
"""

from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional, Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import stft
import pywt


@dataclass
class SpectrogramParamsSeq:
    fs: float = 200.0
    nperseg: int = 256
    noverlap: int = 128
    nfft: Optional[int] = None
    transform: Literal["stft", "cwt"] = "stft"
    # CWT params
    cwt_widths: Optional[np.ndarray] = None  # if None, auto-generate


class IMU2DSequenceSpectrogramDataset(Dataset):
    def __init__(
        self,
        sequences: Sequence[np.ndarray],  # each (C, T)
        labels: Sequence,
        channel_indices: Optional[List[int]] = None,
        spec_params: SpectrogramParamsSeq = SpectrogramParamsSeq(),
        log_amplitude: bool = True,
        eps: float = 1e-6,
    ) -> None:
        assert len(sequences) == len(labels)
        self.seq = [np.asarray(x, dtype=np.float32) for x in sequences]
        self.labels = np.asarray(labels)
        # Default: use all IMU channels
        self.channel_indices = channel_indices if channel_indices is not None else list(range(self.seq[0].shape[0]))

        # ----------------------------
        # Sanity checks
        # ----------------------------
        assert len(self.channel_indices) > 0, "channel_indices cannot be empty"
        max_ch = self.seq[0].shape[0] - 1
        assert all(0 <= ch <= max_ch for ch in self.channel_indices), "channel_indices out of range"
        assert all(seq.shape[0] > max_ch for seq in self.seq), "Some sequences have fewer channels than the largest index in channel_indices"
        # Spectrogram parameter validation
        assert spec_params.fs > 0, "Sampling rate fs must be positive"
        assert spec_params.nperseg > 0, "nperseg must be > 0"
        assert 0 <= spec_params.noverlap < spec_params.nperseg, "noverlap must satisfy 0 <= noverlap < nperseg"
        if spec_params.nfft is not None:
            assert spec_params.nfft >= spec_params.nperseg, "nfft must be >= nperseg"

        self.spec_params = spec_params
        self.log_amplitude = log_amplitude
        self.eps = eps

        # Probe to determine output grid
        if self.spec_params.transform == "stft":
            x0 = self.seq[0][self.channel_indices[0]]
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
            widths = self._get_cwt_widths()
            # cwt returns (len(widths), T)
            self.freq_bins = len(widths)
            self.time_bins = self.seq[0].shape[1]

    def __len__(self) -> int:
        return len(self.labels)

    def _get_cwt_widths(self) -> np.ndarray:
        if self.spec_params.cwt_widths is not None:
            return np.asarray(self.spec_params.cwt_widths, dtype=np.float32)
        # Auto-generate widths to roughly cover frequencies up to fs/2
        # Using Morlet2 wavelet, widths map inversely to frequency. Use geometric progression.
        num_scales = 64
        widths = np.geomspace(1.5, 64.0, num_scales)
        return widths.astype(np.float32)

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
            widths = self._get_cwt_widths()
            coeffs, _ = pywt.cwt(x_1d.astype(np.float32), widths, 'morl', sampling_period=1.0 / self.spec_params.fs)
            mag = np.abs(coeffs).astype(np.float32)  # (scales, T)
        if self.log_amplitude:
            mag = np.log(mag + self.eps)
        return mag

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.seq[idx]  # (C, T)
        spec_list = []
        for ch in self.channel_indices:
            spec = self._spectrogram(x[ch])
            spec_list.append(spec)
        spec = np.stack(spec_list, axis=0)
        y = self.labels[idx]
        return torch.from_numpy(spec), torch.tensor(y)



if __name__ == "__main__":
    # Quick sanity test for IMU2DSequenceSpectrogramDataset
    N, C, T = 5, 6, 512
    rng = np.random.RandomState(1)
    sequences = [rng.randn(C, T).astype(np.float32) for _ in range(N)]
    labels = np.arange(N)

    ds = IMU2DSequenceSpectrogramDataset(sequences, labels)
    spec, lbl = ds[0]
    print("Spec shape:", spec.shape)
    print("Label:", lbl)
    print("Freq bins / Time bins:", ds.freq_bins, ds.time_bins)


