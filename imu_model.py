from __future__ import annotations

"""Baseline IMU-only binary classifier.

This module defines two public symbols:
    - BinaryIMUConfig : A dataclass grouping all hyper-parameters so that no
      magic values live inside the model definition.
    - BinaryIMUCNN    : A lightweight 1-D CNN that leverages Squeeze-and-Excite
      (SE) residual blocks to capture temporal dependencies in IMU streams.

The architecture matches the design described in *Stage 2: Binary
Classification (BFRB vs. non-BFRB)* of `bfrb_detection_pipeline.md` while being
flexible enough for quick experimentation.
"""

from dataclasses import dataclass, field
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from se_resnet_block import ResNetSEBlock  # local util defined in project root


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

@dataclass
class BinaryIMUConfig:
    """Container for all hyper-parameters.

    Modify these values (e.g. through a YAML/JSON loader or argparse) instead
    of hard-coding numbers inside the model.
    """

    # Model topology
    in_channels: int = 20  # Number of IMU features per timestep (acc & rot)
    conv_channels: Sequence[int] = field(
        default_factory=lambda: [32, 64, 128]
    )  # filters for successive Conv blocks
    kernel_size: int = 3  # 1-D convolution kernel size
    use_se: bool = True  # Toggle SE residual blocks vs vanilla conv-BN-ReLU

    # Regularisation & optimisation
    dropout: float = 0.3

    # Classification head
    num_classes: int = 1  # Binary => 1 logit, use BCEWithLogitsLoss

    # Weight initialisation
    kaiming_mode: str = "fan_out"
    kaiming_nonlin: str = "relu"

    def __post_init__(self) -> None:
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd to preserve sequence length with padding")


# ---------------------------------------------------------------------------
#  Model definition
# ---------------------------------------------------------------------------

class BinaryIMUCNN(nn.Module):
    """IMU-only binary classifier with optional SE residual blocks."""

    def __init__(self, cfg: BinaryIMUConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or BinaryIMUConfig()

        layers: List[nn.Module] = []
        in_c = self.cfg.in_channels
        pad = self.cfg.kernel_size // 2  # keep temporal length

        for out_c in self.cfg.conv_channels:
            if self.cfg.use_se:
                layers.append(ResNetSEBlock(in_c, out_c))
            else:
                layers.extend([
                    nn.Conv1d(in_c, out_c, kernel_size=self.cfg.kernel_size, padding=pad, bias=False),
                    nn.BatchNorm1d(out_c),
                    nn.ReLU(inplace=True),
                ])
            layers.append(nn.Dropout(self.cfg.dropout))
            in_c = out_c

        self.feature_extractor = nn.Sequential(*layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)  # (B, C, 1)
        self.fc = nn.Linear(in_c, self.cfg.num_classes)

        self._init_weights()

    # ---------------------------------------------------------------------
    #  Forward pass
    # ---------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        x = self.feature_extractor(x)
        x = self.global_pool(x).squeeze(-1)  # (B, C)
        logits = self.fc(x)
        return logits  # raw logits; apply sigmoid during inference if needed

    # ---------------------------------------------------------------------
    #  Utilities
    # ---------------------------------------------------------------------

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy with logits (handles class weights if given)."""
        return F.binary_cross_entropy_with_logits(logits.view_as(targets), targets.float())

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode=self.cfg.kaiming_mode, nonlinearity=self.cfg.kaiming_nonlin
                )
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


# ---------------------------------------------------------------------------
#  Quick sanity test (executed only when running this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = BinaryIMUConfig()
    model = BinaryIMUCNN(cfg)
    dummy = torch.randn(4, cfg.in_channels, 300)  # (batch, channels, timesteps)
    logits = model(dummy)
    print("output shape:", logits.shape)  # Expect (4, 1)