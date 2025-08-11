from __future__ import annotations

"""Multimodal gesture classifier combining IMU, THM, TOF branches.

Pipeline
--------
1. Modality-specific encoders produce latent vectors of identical dimension.
2. Latents are stacked as a length-3 sequence and processed by a small
   Transformer encoder so each modality can attend to the others.
3. Pooled representation (mean over sequence) feeds the final MLP head.

This mirrors the Stage-3 design described in *bfrb_detection_pipeline.md* with
an additional self-attention fusion step.
"""

from dataclasses import dataclass, field
from typing import Tuple, List

import torch
import torch.nn as nn

from imu_model import BinaryIMUCNN, BinaryIMUConfig
from thm_model import THMGRU, THMGRUConfig
from tof_model import TOFEfficientNet, TOFEfficientNetConfig

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------


@dataclass
class MultiModalConfig:
    """Hyper-parameters for the multimodal classifier."""

    imu_cfg: BinaryIMUConfig = field(default_factory=BinaryIMUConfig)
    thm_cfg: THMGRUConfig = field(default_factory=THMGRUConfig)
    tof_cfg: TOFEfficientNetConfig = field(default_factory=TOFEfficientNetConfig)

    # Transformer fusion
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1

    # Classification head
    num_classes: int = 17  # change to dataset gesture count


# ---------------------------------------------------------------------------
#  Model
# ---------------------------------------------------------------------------


class MultiModalTransformerClassifier(nn.Module):
    """Fuse IMU/THM/TOF latent vectors with a Transformer encoder."""

    def __init__(self, cfg: MultiModalConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MultiModalConfig()

        # --- Modality branches ---
        self.imu_encoder = BinaryIMUCNN(self.cfg.imu_cfg)
        self.thm_encoder = THMGRU(self.cfg.thm_cfg)
        self.tof_encoder = TOFEfficientNet(self.cfg.tof_cfg)

        latent_dim = self.cfg.imu_cfg.conv_channels[-1]  # IMU output dim
        assert (
            self.cfg.thm_cfg.latent_dim == latent_dim == self.cfg.tof_cfg.latent_dim
        ), "All branch latent dims must match"

        # Positional encoding for 3-token sequence (learnable)
        self.pos_emb = nn.Parameter(torch.zeros(3, latent_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=self.cfg.n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=self.cfg.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(latent_dim, self.cfg.num_classes),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, imu: torch.Tensor, thm: torch.Tensor, tof: torch.Tensor) -> torch.Tensor:
        imu_latent = self.imu_encoder.feature_extractor(imu)
        imu_latent = self.imu_encoder.global_pool(imu_latent).squeeze(-1)  # (B, D)

        thm_latent = self.thm_encoder(thm)  # (B, D)
        tof_latent = self.tof_encoder(tof)  # (B, D)

        # Stack as sequence (B, 3, D) and add positional embeddings
        seq = torch.stack([imu_latent, thm_latent, tof_latent], dim=1)  # (B, 3, D)
        seq = seq + self.pos_emb  # broadcast over batch

        fused = self.transformer(seq)  # (B, 3, D)
        pooled = fused.mean(dim=1)  # simple average pooling over tokens

        logits = self.classifier(pooled)  # (B, num_classes)
        return logits

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        nn.init.normal_(self.pos_emb, std=0.02)


# ---------------------------------------------------------------------------
#  Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = MultiModalConfig()
    model = MultiModalTransformerClassifier(cfg)
    B = 2
    imu_dummy = torch.randn(B, cfg.imu_cfg.in_channels, 300)
    thm_dummy = torch.randn(B, 120, cfg.thm_cfg.input_dim)
    tof_dummy = torch.randn(B, 5, 8, 8)
    logits = model(imu_dummy, thm_dummy, tof_dummy)
    print("logits", logits.shape)