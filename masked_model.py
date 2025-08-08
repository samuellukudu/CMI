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
    # Increased depth for richer feature hierarchy in the TOF decoder
    tof_conv_channels: Sequence[int] = field(default_factory=lambda: [256, 128, 64, 32])  # Reverse of encoder
    tof_initial_length: int = 10  # Initial sequence length for TOF decoder
    tof_upsample_factor: int = 2  # Upsampling factor per layer

    # Loss configuration
    loss_type: str = "mae"  # options: "mae", "huber"
    huber_beta: float = 1.0  # delta parameter for Huber/SmoothL1 loss

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

        # -------- confidence-aware uncertainty parameters --------
        if self.cfg.loss_type == "conf_mse":
            # Learnable log variance parameters for THM and TOF heads
            self.log_var_thm = nn.Parameter(torch.zeros(()))
            self.log_var_tof = nn.Parameter(torch.zeros(()))

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

    def reconstruction_loss(
        self,
        preds: Tuple[torch.Tensor, torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor],
        masks: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, float, float]:
        """Compute loss on masked positions only.
        
        Returns:
            Tuple of (total_loss, thm_loss_item, tof_loss_item)
        """
        thm_pred, tof_pred = preds
        thm_tgt, tof_tgt = targets
        thm_mask, tof_mask = masks  # 0/1 floats (1 => was masked)

        # Confidence-aware MSE loss branch
        if self.cfg.loss_type == "conf_mse":
            mse_fn = nn.MSELoss(reduction="none")

            def _masked_mse(pred: torch.Tensor, tgt: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
                mse = mse_fn(pred, tgt) * m
                denom = m.sum().clamp(min=1.0)
                return mse.sum() / denom

            mse_thm = _masked_mse(thm_pred, thm_tgt, thm_mask)
            mse_tof = _masked_mse(tof_pred, tof_tgt, tof_mask)

            var_thm = torch.exp(self.log_var_thm)
            var_tof = torch.exp(self.log_var_tof)

            loss_thm = 0.5 * mse_thm / var_thm + 0.5 * self.log_var_thm
            loss_tof = 0.5 * mse_tof / var_tof + 0.5 * self.log_var_tof

            # Balance contributions by inverse output dims
            w_thm = 1.0 / thm_pred.size(-1)
            w_tof = 1.0 / tof_pred.size(-1)
            total_loss = w_thm * loss_thm + w_tof * loss_tof
            return total_loss, loss_thm.item(), loss_tof.item()

        # Choose loss function based on configuration
        if self.cfg.loss_type == "huber":
            criterion = nn.SmoothL1Loss(beta=self.cfg.huber_beta, reduction="none")
        else:
            criterion = nn.L1Loss(reduction="none")

        def _masked_loss(pred: torch.Tensor, tgt: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            loss = criterion(pred, tgt) * m
            # Avoid division by zero
            denom = m.sum().clamp(min=1.0)
            return loss.sum() / denom

        loss_thm = _masked_loss(thm_pred, thm_tgt, thm_mask)
        loss_tof = _masked_loss(tof_pred, tof_tgt, tof_mask)

        # Balance contributions by scaling with inverse output dimensions
        w_thm = 1.0 / thm_pred.size(-1)  # 1 / 17
        w_tof = 1.0 / tof_pred.size(-1)  # 1 / 320
        total_loss = w_thm * loss_thm + w_tof * loss_tof
        
        return total_loss, loss_thm.item(), loss_tof.item()

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