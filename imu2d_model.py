from __future__ import annotations

"""IMU 2D spectrogram classifier using EfficientNet from timm.

This module defines:
    - IMU2DEfficientNetConfig : dataclass holding EfficientNet hyper-parameters
    - IMU2DEfficientNet       : classifier that adapts IMU spectrograms to an
      EfficientNet backbone and outputs gesture logits.

Input shape: (B, C, F, T)
  - C: spectrogram channels (e.g. 3 for acc_x/acc_y/acc_z)
  - F: frequency bins
  - T: time bins
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # type: ignore


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

@dataclass
class IMU2DEfficientNetConfig:
    """Hyper-parameters for the EfficientNet spectrogram classifier."""

    model_name: str = "efficientnet_b0"  # any timm EfficientNet variant
    pretrained: bool = True
    # Target spatial resolution for spectrogram images expected by EfficientNet
    # Can be either a single integer (square resize) or a (height, width) tuple for
    # non-square resizing.
    img_size: int | tuple[int, int] = 224  # EfficientNet default resolution

    # Input / preprocessing
    in_channels: int = 3  # spectrogram channels
    compress_to3: bool = True  # map arbitrary C→3 with 1×1 conv for ImageNet weights

    # Output
    num_classes: int = 18  # 18 gestures
    dropout: float = 0.2  # dropout before final linear layer
    # Optionally freeze EfficientNet backbone during training
    freeze_backbone: bool = True


# ---------------------------------------------------------------------------
#  Model definition
# ---------------------------------------------------------------------------

class IMU2DEfficientNet(nn.Module):
    """Spectrogram classifier built on an EfficientNet backbone."""

    def __init__(self, cfg: IMU2DEfficientNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or IMU2DEfficientNetConfig()

        # Optional 1×1 conv to map spectrogram channels → 3 so we can reuse ImageNet weights
        in_chans_backbone = 3 if (self.cfg.compress_to3 and self.cfg.in_channels != 3) else self.cfg.in_channels
        if self.cfg.compress_to3 and self.cfg.in_channels != 3:
            self.channel_mapper = nn.Conv2d(self.cfg.in_channels, 3, kernel_size=1)
        else:
            self.channel_mapper = nn.Identity()

        # timm backbone returning global pooled feature vector
        self.backbone = timm.create_model(
            self.cfg.model_name,
            pretrained=self.cfg.pretrained,
            in_chans=in_chans_backbone,
            num_classes=0,  # remove classifier, outputs (B, feat_dim)
            global_pool="avg",
        )

        # Optionally freeze the EfficientNet backbone to speed up training
        if self.cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        feat_dim = self.backbone.num_features  # type: ignore[attr-defined]

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(self.cfg.dropout),
            nn.Linear(feat_dim, self.cfg.num_classes),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, F, T)
        # Map to 3-channel if necessary
        x = self.channel_mapper(x)

        # Resize spectrogram to the target resolution expected by EfficientNet
        if isinstance(self.cfg.img_size, int):
            target_hw = (self.cfg.img_size, self.cfg.img_size)
        else:
            target_hw = self.cfg.img_size  # (H, W)

        if x.shape[-2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)

        feats = self.backbone(x)  # (B, feat_dim)
        logits = self.head(feats)  # (B, num_classes)
        return logits

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Kaiming-uniform init for the classification head."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
#  Quick sanity test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = IMU2DEfficientNetConfig()
    model = IMU2DEfficientNet(cfg)
    dummy = torch.randn(4, cfg.in_channels, 129, 64)
    out = model(dummy)
    print("output", out.shape)