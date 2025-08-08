"""Masked sensor reconstruction model (Stage-1 imputation).

Architecture overview
---------------------
Encoder
    • Takes IMU features shaped *(B, C_imu, L)* — **L can be 1** if working
      on aggregated features.
    • Stacked ResNet-SE blocks (see `se_resnet_block.py`).
    • Global-average-pooled to a latent vector.
Decoder(s)
    • Two independent MLP branches to reconstruct THM *(17 dims)* and TOF
      *(n_tof dims, e.g. 320)*.

Only the masked positions (ratio 0.5 – 0.8) contribute to the loss, see
`SensorMaskedDataset` for mask generation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from se_resnet_block import ResNetSEBlock

# ---------------------------------------------------------------------------
#  Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class MaskedImputerConfig:
    """Hyper-parameters for the MaskedImputer model."""

    # Encoder
    in_channels_imu: int = 20  # IMU feature count
    conv_channels: Sequence[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_size: int = 3  # 1-D conv kernel (odd => length preserved)

    # Latent & decoders
    latent_dim: int = 256
    thm_out_dim: int = 17
    tof_out_dim: int = 320  # 5×8×8 flattened grid default
    
    # TOF transposed conv decoder
    tof_conv_channels: Sequence[int] = field(default_factory=lambda: [128, 64, 32])  # Reverse of encoder
    tof_initial_length: int = 10  # Initial sequence length for TOF decoder
    tof_upsample_factor: int = 2  # Upsampling factor per layer

    # Regularisation
    dropout: float = 0.2

    # Initialisation
    kaiming_mode: str = "fan_out"
    kaiming_nonlin: str = "relu"

    def __post_init__(self) -> None:  # noqa: D401
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd to preserve length")


# ---------------------------------------------------------------------------
#  Model definition
# ---------------------------------------------------------------------------


class MaskedImputer(nn.Module):
    """UNet-like (encoder-decoder) model for sensor value imputation."""

    def __init__(self, cfg: MaskedImputerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MaskedImputerConfig()

        # ----- encoder -----
        layers = []
        in_c = self.cfg.in_channels_imu
        pad = self.cfg.kernel_size // 2
        for out_c in self.cfg.conv_channels:
            layers.append(ResNetSEBlock(in_c, out_c))
            layers.append(nn.Dropout(self.cfg.dropout))
            in_c = out_c
        self.encoder = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.to_latent = nn.Linear(in_c, self.cfg.latent_dim)

        # ----- decoders -----
        # THM decoder with 3 layers for better feature learning
        self.decoder_thm = nn.Sequential(
            nn.Linear(self.cfg.latent_dim, self.cfg.latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.latent_dim // 2, self.cfg.latent_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.latent_dim // 4, self.cfg.thm_out_dim),
        )
        
        # TOF decoder using 1D transposed convolutions
        self._build_tof_decoder()

        self._init_weights()
    
    def _build_tof_decoder(self) -> None:
        """Build TOF decoder using 1D transposed convolutions."""
        # Project latent to initial conv feature map
        initial_channels = self.cfg.tof_conv_channels[0]
        self.tof_latent_proj = nn.Linear(
            self.cfg.latent_dim, 
            initial_channels * self.cfg.tof_initial_length
        )
        
        # Build transposed conv layers
        tof_layers = []
        in_channels = initial_channels
        
        for i, out_channels in enumerate(self.cfg.tof_conv_channels[1:]):
            # Transposed conv with stride=2 for upsampling
            tof_layers.extend([
                nn.ConvTranspose1d(
                    in_channels, out_channels, 
                    kernel_size=self.cfg.kernel_size,
                    stride=self.cfg.tof_upsample_factor,
                    padding=self.cfg.kernel_size // 2,
                    output_padding=1
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(self.cfg.dropout)
            ])
            in_channels = out_channels
        
        # Final layer to get exact TOF dimensions
        tof_layers.append(
            nn.ConvTranspose1d(
                in_channels, 1,
                kernel_size=self.cfg.kernel_size,
                stride=1,
                padding=self.cfg.kernel_size // 2
            )
        )
        
        self.decoder_tof = nn.Sequential(*tof_layers)
        
        # Adaptive pooling to ensure exact output size
        self.tof_adaptive_pool = nn.AdaptiveAvgPool1d(self.cfg.tof_out_dim)

    # ---------------------------------------------------------------------
    #  Forward + loss
    # ---------------------------------------------------------------------

    def forward(self, imu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: D401
        """Return predicted (thm, tof)."""
        # Accept (B, C) or (B, C, L)
        if imu.dim() == 2:
            imu = imu.unsqueeze(-1)  # (B, C, 1)
        z = self.encoder(imu)               # (B, C*, L)
        z = self.global_pool(z).squeeze(-1)  # (B, C*)
        z = self.to_latent(z)               # (B, latent)
        
        # THM prediction (unchanged)
        thm_pred = self.decoder_thm(z)      # (B, thm_out_dim)
        
        # TOF prediction using transposed convolutions
        batch_size = z.size(0)
        # Project to initial feature map
        tof_features = self.tof_latent_proj(z)  # (B, initial_channels * initial_length)
        tof_features = tof_features.view(
            batch_size, 
            self.cfg.tof_conv_channels[0], 
            self.cfg.tof_initial_length
        )  # (B, initial_channels, initial_length)
        
        # Apply transposed convolutions
        tof_conv_out = self.decoder_tof(tof_features)  # (B, 1, upsampled_length)
        
        # Ensure exact output dimensions and flatten
        # Handle MPS device limitation with adaptive pooling
        if tof_conv_out.device.type == 'mps' and tof_conv_out.size(-1) % self.cfg.tof_out_dim != 0:
            # Fallback: use interpolation for MPS when sizes aren't divisible
            tof_pred = F.interpolate(tof_conv_out, size=self.cfg.tof_out_dim, mode='linear', align_corners=False)
            tof_pred = tof_pred.squeeze(1)  # (B, tof_out_dim)
        else:
            tof_pred = self.tof_adaptive_pool(tof_conv_out)  # (B, 1, tof_out_dim)
            tof_pred = tof_pred.squeeze(1)  # (B, tof_out_dim)
        
        return thm_pred, tof_pred

    # ------------------------------------------------------------------
    #  Masked reconstruction loss
    # ------------------------------------------------------------------

    @staticmethod
    def reconstruction_loss(
        preds: Tuple[torch.Tensor, torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor],
        masks: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute MAE on masked positions only."""
        thm_pred, tof_pred = preds
        thm_tgt, tof_tgt = targets
        thm_mask, tof_mask = masks  # 0/1 floats (1 => was masked)

        def _masked_mae(pred: torch.Tensor, tgt: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            mae = torch.abs(pred - tgt) * m
            # Avoid division by zero
            denom = m.sum().clamp(min=1.0)
            return mae.sum() / denom

        loss_thm = _masked_mae(thm_pred, thm_tgt, thm_mask)
        loss_tof = _masked_mae(tof_pred, tof_tgt, tof_mask)
        return loss_thm + loss_tof

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, mode=self.cfg.kaiming_mode, nonlinearity=self.cfg.kaiming_nonlin)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
#  Quick sanity test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = MaskedImputerConfig()
    model = MaskedImputer(cfg)
    B = 4
    dummy_imu = torch.randn(B, cfg.in_channels_imu)  # (B, C)
    thm, tof = model(dummy_imu)
    print("thm:", thm.shape, "tof:", tof.shape)