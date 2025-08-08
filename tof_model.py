"""TOF branch based on pretrained EfficientNet from timm.

This module defines two public symbols:
    - TOFEfficientNetConfig : dataclass holding hyper-parameters
    - TOFEfficientNet       : PyTorch module that adapts small 5×8×8 TOF
      tensors to a standard EfficientNet backbone.

The class expects input *x* shaped (B, 5, 8, 8) where 5 is the number of TOF
frames concatenated as channels. It first compresses the 5-channel tensor to
3 channels with a 1×1 convolution, upsamples to *img_size×img_size* (default
224), and feeds it to a timm EfficientNet variant. The pooled feature vector
is projected down to *latent_dim* for downstream fusion.

Example
-------
>>> cfg = TOFEfficientNetConfig()
>>> model = TOFEfficientNet(cfg)
>>> dummy = torch.randn(4, 5, 8, 8)  # (batch, C, H, W)
>>> feats = model(dummy)
>>> print(feats.shape)  # torch.Size([4, cfg.latent_dim])
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # type: ignore


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

@dataclass
class TOFEfficientNetConfig:
    """Hyper-parameters for the TOF EfficientNet branch."""

    model_name: str = "efficientnet_b0"  # any timm EfficientNet variant
    pretrained: bool = True
    img_size: int = 224  # efficientnet default resolution
    latent_dim: int = 128  # output feature dimension

    # Pre-processing
    in_channels: int = 5  # TOF frames stacked along channel axis
    compress_to3: bool = True  # use 1×1 conv to convert to 3-channel

    # Weight init for the projection layer
    kaiming_mode: str = "fan_out"
    kaiming_nonlin: str = "relu"


# ---------------------------------------------------------------------------
#  Model definition
# ---------------------------------------------------------------------------

class TOFEfficientNet(nn.Module):
    """TOF feature extractor built on EfficientNet."""

    def __init__(self, cfg: TOFEfficientNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TOFEfficientNetConfig()

        # Optional 1×1 conv to map in_channels→3 so we can reuse ImageNet weights
        in_chans_backbone = 3 if self.cfg.compress_to3 else self.cfg.in_channels
        if self.cfg.compress_to3 and self.cfg.in_channels != 3:
            self.channel_mapper = nn.Conv2d(self.cfg.in_channels, 3, kernel_size=1)
        else:
            self.channel_mapper = nn.Identity()

        # timm backbone returning the final global pooled feature vector
        self.backbone = timm.create_model(
            self.cfg.model_name,
            pretrained=self.cfg.pretrained,
            in_chans=in_chans_backbone,
            num_classes=0,  # removes the classification head, outputs (B, feat_dim)
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features  # type: ignore[attr-defined]

        # Projection to latent_dim used by downstream fusion layers
        self.proj = nn.Linear(feat_dim, self.cfg.latent_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 5, 8, 8)
        if x.dim() != 4:
            raise ValueError("Expected 4-D tensor (B, C=5, H=8, W=8)")

        # Map to 3 channels if necessary
        x = self.channel_mapper(x)  # (B, 3, 8, 8) or identity

        # Upsample to EfficientNet input size
        if x.shape[-1] != self.cfg.img_size:
            x = F.interpolate(x, size=(self.cfg.img_size, self.cfg.img_size), mode="bilinear", align_corners=False)

        feats = self.backbone(x)  # (B, feat_dim)
        latent = self.proj(feats)  # (B, latent_dim)
        return latent

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(
            self.proj.weight, mode=self.cfg.kaiming_mode, nonlinearity=self.cfg.kaiming_nonlin
        )
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)


# ---------------------------------------------------------------------------
#  Quick sanity test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = TOFEfficientNetConfig()
    model = TOFEfficientNet(cfg)
    dummy = torch.randn(2, cfg.in_channels, 8, 8)
    out = model(dummy)
    print("output shape:", out.shape)  # Expect (2, latent_dim)