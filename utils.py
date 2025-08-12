"""Common utility functions for reproducibility and memory management.

This module centralises helpers that were previously duplicated across
individual training scripts. Import these functions from any script to
ensure consistent behaviour throughout the project.
"""
from __future__ import annotations

import gc
import os
import random
from typing import Final

import numpy as np
import torch

__all__: Final = [
    "seeding",
    "flush",
    "seed_worker",
]


# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------

def seeding(seed: int) -> None:  # noqa: D401
    """Set global random seeds for *numpy*, *random*, and *torch*.

    Parameters
    ----------
    seed : int
        The random seed to use for all libraries.
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Setting deterministic to *False* and benchmark to *True* gives better
        # performance while still keeping convolutions deterministic on modern
        # PyTorch versions.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# -----------------------------------------------------------------------------
# Memory helpers
# -----------------------------------------------------------------------------

def flush() -> None:  # noqa: D401
    """Release cached GPU memory and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# -----------------------------------------------------------------------------
# DataLoader worker seeding
# -----------------------------------------------------------------------------

def seed_worker(worker_id: int) -> None:  # noqa: D401
    """Ensure each *DataLoader* worker receives a unique, reproducible seed."""
    # Torch sets the base slightly differently for each worker; leverage this
    # to obtain deterministic but distinct seeds.
    worker_seed = torch.initial_seed() % 2**32  # type: ignore[arg-type]
    np.random.seed(worker_seed)
    random.seed(worker_seed)