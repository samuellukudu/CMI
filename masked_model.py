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
    kernel_size: int = 17  # 1-D conv kernel (odd => length preserved)

    # Mask-aware conditioning (concatenate masked THM/TOF with IMU)
    use_mask_conditioning: bool = False

    # Latent & decoders
    latent_dim: int = 256
    thm_out_dim: int = 17
    tof_out_dim: int = 320  # 5×8×8 flattened grid default
    
    # TOF transposed conv decoder
    # Increased depth for richer feature hierarchy in the TOF decoder
    tof_conv_channels: Sequence[int] = field(default_factory=lambda: [256, 128, 64, 32])  # Reverse of encoder
    tof_initial_length: int = 10  # Initial sequence length for TOF decoder
    tof_upsample_factor: int = 2  # Upsampling factor per layer

    # Shared transformer decoder (optional)
    transformer_nhead: int = 8
    transformer_ffn_mult: int = 4

    # Multi-task learning configuration
    use_shared_decoder: bool = False  # Whether to use shared decoder layers
    shared_decoder_layers: int = 2  # Number of shared decoder layers before branching
    shared_decoder_depth: int = 3  # Depth of shared MLP trunk layers
    use_task_attention: bool = True  # Cross-attention between THM and TOF tasks
    
    # Advanced loss configuration
    loss_type: str = "adaptive_weighted"  # "mae", "huber", "adaptive_weighted", "gradient_cos", "uncertainty_weighted"
    thm_loss_type: str | None = None  # Optional override for THM loss (defaults to loss_type)
    tof_loss_type: str | None = None  # Optional override for TOF loss (defaults to loss_type)
    huber_beta: float = 1.0  # delta parameter for Huber/SmoothL1 loss
    use_uncertainty_weighting: bool = True  # Learn uncertainty-based task weights
    use_homoscedastic_uncertainty: bool = True  # Model output uncertainty
    loss_temperature: float = 1.0  # Temperature for loss balancing
    
    # Adaptive loss weighting
    adaptive_loss_alpha: float = 0.16  # Learning rate for loss weight adaptation
    gradient_normalization: bool = True  # Enable gradient normalization
    
    # Cross-modal alignment
    use_cross_modal_loss: bool = True  # Enable cross-modal consistency loss
    cross_modal_weight: float = 0.1  # Weight for cross-modal loss

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
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.to_latent = nn.Linear(in_c, self.cfg.latent_dim)

        # ----- decoders -----
        if self.cfg.use_shared_decoder:
            # Lightweight shared MLP trunk with task-specific heads
            # Much more efficient than transformer decoder with 337 queries
            shared_hidden = self.cfg.latent_dim // 2
            
            # Shared feature extraction trunk (configurable depth)
            trunk_layers = [
                nn.Linear(self.cfg.latent_dim, shared_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(self.cfg.dropout),
            ]
            for _ in range(max(1, self.cfg.shared_decoder_depth - 1)):
                trunk_layers += [
                    nn.Linear(shared_hidden, shared_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.cfg.dropout),
                ]
            self.shared_trunk = nn.Sequential(*trunk_layers)
            
            # Task-specific heads (increase depth slightly for TOF)
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
                nn.Linear(shared_hidden, shared_hidden // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(shared_hidden // 2, self.cfg.tof_out_dim),
            )
        else:
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

        # -------- confidence-aware uncertainty parameters --------
        if self.cfg.loss_type in ["conf_mse", "conf_mae", "conf_huber", "uncertainty_weighted"]:
            # Learnable log variance/scale parameters for THM and TOF heads
            self.log_var_thm = nn.Parameter(torch.zeros(()))
            self.log_var_tof = nn.Parameter(torch.zeros(()))

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

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5)) if hasattr(nn.init, 'kaiming_uniform_') else None
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
        # Init shared query embeddings if present
        if hasattr(self, 'shared_query_embed'):
            nn.init.normal_(self.shared_query_embed, mean=0.0, std=0.02)

    # ---------------------------------------------------------------------
    #  Forward + loss
    # ---------------------------------------------------------------------

    def forward(self, imu: torch.Tensor, thm_input: torch.Tensor | None = None, tof_input: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: D401
        """Return predicted (thm, tof).
        If cfg.use_mask_conditioning is True, expects thm_input and tof_input (masked copies).
        """
        # Prepare input features
        if self.cfg.use_mask_conditioning:
            if thm_input is None or tof_input is None:
                raise ValueError("use_mask_conditioning=True but thm_input/tof_input not provided")
            # Accept (B, C) or (B, C, L) for each; unify to (B, C, L)
            if imu.dim() == 2:
                imu = imu.unsqueeze(-1)
            if thm_input.dim() == 2:
                thm_input = thm_input.unsqueeze(-1)
            if tof_input.dim() == 2:
                tof_input = tof_input.unsqueeze(-1)
            x = torch.cat([imu, thm_input, tof_input], dim=1)
        else:
            # Accept (B, C) or (B, C, L)
            x = imu if imu.dim() == 3 else imu.unsqueeze(-1)

        z = self.encoder(x)               # (B, C*, L)
        z = self.global_pool(z).squeeze(-1)  # (B, C*)
        z = self.to_latent(z)               # (B, latent)
        
        if self.cfg.use_shared_decoder:
            # Shared MLP trunk
            shared_feat = self.shared_trunk(z)  # (B, shared_hidden)
            thm_pred = self.decoder_thm(shared_feat)
            tof_pred = self.decoder_tof(shared_feat)
            return thm_pred, tof_pred
        
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
        if tof_conv_out.device.type == 'mps' and tof_conv_out.size(-1) % self.cfg.tof_out_dim != 0:
            tof_out = F.interpolate(tof_conv_out, size=self.cfg.tof_out_dim, mode='linear', align_corners=False)
        else:
            tof_out = self.tof_adaptive_pool(tof_conv_out)
        
        tof_pred = tof_out.squeeze(1)  # (B, tof_out_dim)
        return thm_pred, tof_pred

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

        # Determine per-task loss types (allow overrides)
        thm_loss_type = (self.cfg.thm_loss_type or self.cfg.loss_type).lower()
        tof_loss_type = (self.cfg.tof_loss_type or self.cfg.loss_type).lower()

        # Helper to get criterion by name
        def get_criterion(name: str):
            if name == "huber":
                return nn.SmoothL1Loss(beta=self.cfg.huber_beta, reduction="none")
            elif name == "mae":
                return nn.L1Loss(reduction="none")
            elif name == "mse":
                return nn.MSELoss(reduction="none")
            else:
                # For composite types, base criterion will be MAE; combination handled later
                return nn.L1Loss(reduction="none")

        # Base masked loss (without composite weighting)
        def masked_loss_with(criterion, pred, tgt, m):
            loss = criterion(pred, tgt) * m
            denom = m.sum().clamp(min=1.0)
            return loss.sum() / denom

        # Compute base per-task losses
        base_loss_thm = masked_loss_with(get_criterion(thm_loss_type), thm_pred, thm_tgt, thm_mask)
        base_loss_tof = masked_loss_with(get_criterion(tof_loss_type), tof_pred, tof_tgt, tof_mask)

        loss_thm = base_loss_thm
        loss_tof = base_loss_tof

        # Advanced loss weighting strategies (by global loss_type)
        lt = self.cfg.loss_type
        if lt == "uncertainty_weighted":
            precision_thm = torch.exp(-self.log_var_thm)
            precision_tof = torch.exp(-self.log_var_tof)
            weighted_loss_thm = precision_thm * loss_thm + self.log_var_thm
            weighted_loss_tof = precision_tof * loss_tof + self.log_var_tof
            total_loss = weighted_loss_thm + weighted_loss_tof
        elif lt == "adaptive_weighted":
            if not hasattr(self, 'task_weights'):
                self.task_weights = nn.Parameter(torch.tensor([1.0, 1.0], requires_grad=True))
            weights = F.softmax(self.task_weights, dim=0) * 2.0
            total_loss = weights[0] * loss_thm + weights[1] * loss_tof
        elif lt == "gradient_cos":
            if self.training:
                loss_thm.backward(retain_graph=True)
                grad_thm = []
                for p in self.parameters():
                    if p.grad is not None:
                        grad_thm.append(p.grad.clone().flatten())
                self.zero_grad()
                loss_tof.backward(retain_graph=True)
                grad_tof = []
                for p in self.parameters():
                    if p.grad is not None:
                        grad_tof.append(p.grad.clone().flatten())
                self.zero_grad()
                if grad_thm and grad_tof:
                    gthm = torch.cat(grad_thm)
                    gtof = torch.cat(grad_tof)
                    cos_sim = F.cosine_similarity(gthm, gtof, dim=0)
                    alpha = 0.5 + 0.3 * cos_sim
                    beta = 1.0 - alpha
                    total_loss = alpha * loss_thm + beta * loss_tof
                else:
                    total_loss = 0.5 * loss_thm + 0.5 * loss_tof
            else:
                total_loss = 0.5 * loss_thm + 0.5 * loss_tof
        elif lt.startswith("conf_"):
            base_name = lt[5:]
            # choose error fn
            if base_name == "mse":
                error_fn = nn.MSELoss(reduction="none")
                def u_loss(e, lv):
                    var = torch.exp(lv)
                    return 0.5 * e / var + 0.5 * lv
            elif base_name == "mae":
                error_fn = nn.L1Loss(reduction="none")
                def u_loss(e, lv):
                    b = torch.exp(lv)
                    return e / b + lv + 0.693
            else:  # huber
                beta = self.cfg.huber_beta
                error_fn = nn.SmoothL1Loss(beta=beta, reduction="none")
                def u_loss(e, lv):
                    scale = torch.exp(lv)
                    return e / scale + lv
            def masked_error(pred, tgt, m):
                err = error_fn(pred, tgt) * m
                denom = m.sum().clamp(min=1.0)
                return err.sum() / denom
            e_thm = masked_error(thm_pred, thm_tgt, thm_mask)
            e_tof = masked_error(tof_pred, tof_tgt, tof_mask)
            loss_thm = u_loss(e_thm, self.log_var_thm)
            loss_tof = u_loss(e_tof, self.log_var_tof)
            w_thm = 1.0 / thm_pred.size(-1)
            w_tof = 1.0 / tof_pred.size(-1)
            total_loss = w_thm * loss_thm + w_tof * loss_tof
            return total_loss, loss_thm.item(), loss_tof.item()
        else:
            w_thm = 1.0 / thm_pred.size(-1)
            w_tof = 1.0 / tof_pred.size(-1)
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
        # Initialize shared decoder query embeddings if present
        if hasattr(self, 'shared_query_embed'):
            nn.init.normal_(self.shared_query_embed, std=0.02)


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