"""THM branch based on a two-layer bidirectional GRU.

Public API
----------
• THMGRUConfig – hyper-parameters container.
• THMGRU       – PyTorch module that produces a latent vector from THM
  feature sequences shaped (B, T, F).

A quick sanity check is included under the __main__ guard.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

@dataclass
class THMGRUConfig:
    """Hyper-parameters for the THM bidirectional GRU."""

    input_dim: int = 17          # number of THM features per timestep
    hidden_dim: int = 64         # GRU hidden size (per direction)
    num_layers: int = 2          # stacked GRU layers
    bidirectional: bool = True   # use Bi-GRU
    dropout: float = 0.3         # dropout between GRU layers
    latent_dim: int = 128        # projection dimension for downstream fusion

    # Weight init for the projection layer
    kaiming_mode: str = "fan_out"
    kaiming_nonlin: str = "relu"


# ---------------------------------------------------------------------------
#  Model
# ---------------------------------------------------------------------------

class THMGRU(nn.Module):
    """Extract latent representation from THM sequences using Bi-GRU."""

    def __init__(self, cfg: THMGRUConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or THMGRUConfig()

        self.gru = nn.GRU(
            input_size=self.cfg.input_dim,
            hidden_size=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            batch_first=True,
            dropout=self.cfg.dropout if self.cfg.num_layers > 1 else 0.0,
            bidirectional=self.cfg.bidirectional,
        )

        # Calculate feature dimension after GRU
        feat_dim = self.cfg.hidden_dim * (2 if self.cfg.bidirectional else 1)
        self.proj = nn.Linear(feat_dim, self.cfg.latent_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F)
        if x.dim() != 3:
            raise ValueError("Expected 3-D tensor (B, T, F)")

        outputs, h_n = self.gru(x)  # outputs: (B, T, D*)
        # Use the last timestep from outputs as representation
        last_out = outputs[:, -1, :]  # (B, D*)
        latent = self.proj(last_out)  # (B, latent_dim)
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
    cfg = THMGRUConfig()
    model = THMGRU(cfg)
    dummy = torch.randn(4, 120, cfg.input_dim)  # (batch, seq_len, features)
    out = model(dummy)
    print("output shape:", out.shape)  # Expect (4, latent_dim)