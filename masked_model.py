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
import math

from se_resnet_block import ResNetSEBlock

# ---------------------------------------------------------------------------
#  Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class MaskedImputerConfig:
    """Hyper-parameters for the MaskedImputer model."""

    # Encoder
    in_channels_imu: int = 20  # IMU feature count
    conv_channels: Sequence[int] = field(default_factory=lambda: [64, 128, 256])  # Increased depth
    kernel_size: int = 17  # 1-D conv kernel (odd => length preserved)

    # Mask-aware conditioning (concatenate masked THM/TOF with IMU)
    use_mask_conditioning: bool = False

    # Latent & decoders
    latent_dim: int = 512  # Increased latent capacity
    thm_out_dim: int = 17
    tof_out_dim: int = 320  # 5×8×8 flattened grid default
    imu_out_dim: int = 20  # IMU feature count (same as in_channels_imu)
    # IMU loss configuration
    imu_loss_type: str | None = None  # Optional override for IMU loss
    imu_weight: float = 1.0
    
    # Simplified TOF decoder - no more transposed convs
    tof_hidden_dims: Sequence[int] = field(default_factory=lambda: [512, 256, 128])

    # Multi-task learning configuration
    use_shared_decoder: bool = False  # Whether to use shared decoder layers
    shared_decoder_layers: int = 2  # Number of shared decoder layers before branching
    shared_decoder_depth: int = 3  # Depth of shared MLP trunk layers
    use_task_attention: bool = True  # Cross-attention between THM and TOF tasks
    # UNet-style decoder (shared up-sampling trunk + heads)
    use_unet_decoder: bool = False
    
    # Improved loss configuration with better balancing
    loss_type: str = "balanced_mse"  # New balanced approach
    thm_loss_type: str | None = None  # Optional override for THM loss (defaults to loss_type)
    tof_loss_type: str | None = None  # Optional override for TOF loss (defaults to loss_type)
    huber_beta: float = 1.0  # delta parameter for Huber/SmoothL1 loss
    use_uncertainty_weighting: bool = True  # Learn uncertainty-based task weights
    use_homoscedastic_uncertainty: bool = True  # Model output uncertainty
    loss_temperature: float = 1.0  # Temperature for loss balancing
    
    # Better adaptive loss weighting
    adaptive_loss_alpha: float = 0.1  # Reduced learning rate for stability
    gradient_normalization: bool = True  # Enable gradient normalization
    
    # Cross-modal alignment
    use_cross_modal_loss: bool = False  # Disable for now - adds complexity
    cross_modal_weight: float = 0.1  # Weight for cross-modal loss

    # Regularisation
    dropout: float = 0.1  # Reduced dropout

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
        # Determine input channel size based on mask-aware conditioning
        self.input_channels = self.cfg.in_channels_imu + (
            (self.cfg.thm_out_dim + self.cfg.tof_out_dim) if self.cfg.use_mask_conditioning else 0
        )
        in_c = self.input_channels
        for out_c in self.cfg.conv_channels:
            layers.append(ResNetSEBlock(in_c, out_c))
            layers.append(nn.Dropout(self.cfg.dropout))
            in_c = out_c
        self.encoder = nn.Sequential(*layers)
        
        # Simplified bottleneck - remove complex GRU
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.to_latent = nn.Sequential(
            nn.Linear(in_c, self.cfg.latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout)
        )

        # ----- Decoders -----
        if getattr(self.cfg, 'use_unet_decoder', False):
            # Shared up-sampling trunk that maintains temporal resolution
            self.upsample_trunk = nn.Sequential(
                nn.Conv1d(in_c, in_c // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(self.cfg.dropout),
                nn.Conv1d(in_c // 2, in_c // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(self.cfg.dropout),
            )
            trunk_c = in_c // 2
            self.head_thm = nn.Conv1d(trunk_c, self.cfg.thm_out_dim, kernel_size=1)
            self.head_tof = nn.Conv1d(trunk_c, self.cfg.tof_out_dim, kernel_size=1)
            self.head_imu = nn.Conv1d(trunk_c, self.cfg.imu_out_dim, kernel_size=1)
        elif self.cfg.use_shared_decoder:
            # Lightweight shared MLP trunk with task-specific heads
            shared_hidden = self.cfg.latent_dim // 2

            # Shared trunk
            self.shared_trunk = nn.Sequential(
                nn.Linear(self.cfg.latent_dim, shared_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(shared_hidden, shared_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(self.cfg.dropout),
            )
            
            # Task-specific heads
            self.decoder_thm = nn.Sequential(
                nn.Linear(shared_hidden, shared_hidden // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(shared_hidden // 2, self.cfg.thm_out_dim),
            )
            
            self.decoder_tof = nn.Sequential(
                nn.Linear(shared_hidden, shared_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(shared_hidden, self.cfg.tof_out_dim),
            )
            # IMU decoder head
            self.decoder_imu = nn.Sequential(
                nn.Linear(shared_hidden, shared_hidden // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(shared_hidden // 2, self.cfg.imu_out_dim),
            )
        else:
            # THM decoder - simple and effective
            self.decoder_thm = nn.Sequential(
                nn.Linear(self.cfg.latent_dim, self.cfg.latent_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(self.cfg.latent_dim // 2, self.cfg.latent_dim // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(self.cfg.latent_dim // 4, self.cfg.thm_out_dim),
            )
            
            # TOF decoder - simplified MLP architecture
            tof_layers = []
            in_dim = self.cfg.latent_dim
            for hidden_dim in self.cfg.tof_hidden_dims:
                tof_layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.cfg.dropout)
                ])
                in_dim = hidden_dim
            tof_layers.append(nn.Linear(in_dim, self.cfg.tof_out_dim))
            self.decoder_tof = nn.Sequential(*tof_layers)
            # IMU decoder - mirrors THM branch
            self.decoder_imu = nn.Sequential(
                nn.Linear(self.cfg.latent_dim, self.cfg.latent_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(self.cfg.latent_dim // 2, self.cfg.imu_out_dim),
            )

        self._init_weights()

        # -------- uncertainty parameters for balanced loss --------
        if "balanced" in self.cfg.loss_type or self.cfg.use_uncertainty_weighting:
            # Learnable log variance/scale parameters for THM and TOF heads
            self.log_var_thm = nn.Parameter(torch.zeros(()))
            self.log_var_tof = nn.Parameter(torch.zeros(()))

    def forward(self, imu: torch.Tensor, thm_input: torch.Tensor | None = None, tof_input: torch.Tensor | None = None, imu_input: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # noqa: D401
        """Return predicted (thm, tof).
        If cfg.use_mask_conditioning is True, expects thm_input and tof_input (masked copies).
        """
        # Prepare input features
        if self.cfg.use_mask_conditioning:
            if thm_input is None or tof_input is None or imu_input is None:
                raise ValueError("use_mask_conditioning=True but thm_input/tof_input/imu_input not provided")
            # Accept (B, C) or (B, C, L) for each; unify to (B, C, L)
            if imu.dim() == 2:
                imu = imu.unsqueeze(-1)
            if thm_input.dim() == 2:
                thm_input = thm_input.unsqueeze(-1)
            if tof_input.dim() == 2:
                tof_input = tof_input.unsqueeze(-1)
            x = torch.cat([imu_input, thm_input, tof_input], dim=1)
        else:
            # Accept (B, C) or (B, C, L)
            x = imu if imu.dim() == 3 else imu.unsqueeze(-1)

        enc_feat = self.encoder(x)  # (B, C*, L)
        # ---- UNet branch ----
        if getattr(self.cfg, 'use_unet_decoder', False):
            feat = self.upsample_trunk(enc_feat)  # (B, trunk_c, L)
            thm_pred = self.head_thm(feat).mean(dim=-1)  # (B, thm_out)
            tof_pred = self.head_tof(feat).mean(dim=-1)  # (B, tof_out)
            imu_pred = self.head_imu(feat).mean(dim=-1)  # (B, imu_out)
            return thm_pred, tof_pred, imu_pred
        # ---- MLP branches ----
        z = self.global_pool(enc_feat).squeeze(-1)  # (B, C*)
        z = self.to_latent(z)  # (B, latent)
        
        if self.cfg.use_shared_decoder:
            shared_feat = self.shared_trunk(z)
            thm_pred = self.decoder_thm(shared_feat)
            tof_pred = self.decoder_tof(shared_feat)
            imu_pred = self.decoder_imu(shared_feat)
            return thm_pred, tof_pred, imu_pred
        
        # Individual decoders
        thm_pred = self.decoder_thm(z)
        tof_pred = self.decoder_tof(z)
        imu_pred = self.decoder_imu(z)
        return thm_pred, tof_pred, imu_pred

    def reconstruction_loss(
        self,
        preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        masks: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, float, float, float]:
        """Compute loss on masked positions only with per-task criteria and balanced aggregation.

        Per-task base losses can be set via cfg.thm_loss_type and cfg.tof_loss_type
        (choices: 'mse', 'mae', 'huber'). If unset, they inherit from a base
        loss inferred from cfg.loss_type (defaulting to 'mse' when the latter is
        an aggregator like 'balanced_mse' or 'adaptive_weighted').
        """
        thm_pred, tof_pred, imu_pred = preds
        thm_tgt, tof_tgt, imu_tgt = targets
        thm_mask, tof_mask, imu_mask = masks  # 0/1 floats (1 => was masked)

        # Helper to compute masked loss
        def masked_loss(pred, tgt, mask, criterion):
            loss = criterion(pred, tgt)  # (B, features) or (B,)
            # Broadcast mask to match loss shape (faster & simpler)
            if mask.shape != loss.shape:
                mask = mask.view(*mask.shape, *([1] * (loss.dim() - mask.dim()))).expand_as(loss)
            masked_loss = loss * mask
            valid_count = mask.sum()
            if valid_count == 0:  # Edge-case: nothing was masked in this batch
                return torch.tensor(0.0, device=loss.device)
            return masked_loss.sum() / valid_count

        # Map string to criterion (reduction='none')
        def get_criterion(name: str) -> nn.Module:
            name = name.lower()
            if name == 'mse':
                return nn.MSELoss(reduction='none')
            if name == 'mae':
                return nn.L1Loss(reduction='none')
            if name == 'huber':
                return nn.SmoothL1Loss(beta=self.cfg.huber_beta, reduction='none')
            # Fallback
            return nn.MSELoss(reduction='none')

        # Determine base loss name (used if per-task override not provided)
        base_name = self.cfg.loss_type.lower()
        if base_name not in {'mse', 'mae', 'huber'}:
            base_name = 'mse'
        thm_base = self.cfg.thm_loss_type.lower() if self.cfg.thm_loss_type else base_name
        tof_base = self.cfg.tof_loss_type.lower() if self.cfg.tof_loss_type else base_name

        crit_thm = get_criterion(thm_base)
        crit_tof = get_criterion(tof_base)
        imu_base = self.cfg.imu_loss_type.lower() if self.cfg.imu_loss_type else base_name
        crit_imu = get_criterion(imu_base)

        # Compute per-task masked losses
        loss_thm = masked_loss(thm_pred, thm_tgt, thm_mask, crit_thm)
        loss_tof = masked_loss(tof_pred, tof_tgt, tof_mask, crit_tof)
        loss_imu = masked_loss(imu_pred, imu_tgt, imu_mask, crit_imu)

        # Aggregation / balancing across tasks
        if self.cfg.loss_type in {"balanced_mse", "uncertainty_weighted"}:
            # Uncertainty-based weighting (Kendall et al.)
            if hasattr(self, 'log_var_thm') and hasattr(self, 'log_var_tof'):
                precision_thm = torch.exp(-self.log_var_thm)
                precision_tof = torch.exp(-self.log_var_tof)
                total_loss = (
                    precision_thm * loss_thm + self.log_var_thm +
                    precision_tof * loss_tof + self.log_var_tof
                )
            else:
                # Unweighted sum – inputs are already min-max scaled
                total_loss = loss_thm + loss_tof + self.cfg.imu_weight * loss_imu

        elif self.cfg.loss_type == "adaptive_weighted":
            # Learnable weights via softmax
            if not hasattr(self, 'task_weights'):
                self.task_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], requires_grad=True))
            weights = F.softmax(self.task_weights, dim=0) * 3.0
            total_loss = weights[0] * loss_thm + weights[1] * loss_tof + weights[2] * loss_imu

        else:
            # Default: unweighted sum (optionally weight IMU)
            total_loss = loss_thm + loss_tof + self.cfg.imu_weight * loss_imu

        return total_loss, loss_thm.item(), loss_tof.item(), loss_imu.item()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.kaiming_uniform_(
                    m.weight,
                    mode=self.cfg.kaiming_mode,
                    nonlinearity=self.cfg.kaiming_nonlin,
                )
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
        # Initialize shared decoder query embeddings if present
        if hasattr(self, 'shared_query_embed'):
            nn.init.normal_(self.shared_query_embed, std=0.02)

    # ---------------------------------------------------------------------
    #  Helpers
    # ---------------------------------------------------------------------



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