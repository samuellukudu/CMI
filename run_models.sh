#!/usr/bin/env bash
set -Eeuo pipefail

# This script creates/uses a uv-managed virtualenv, installs requirements,
# and runs the training scripts. Comment out any sections you don't want to run.
# Usage: bash run_models.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "[1/4] Checking for 'uv'..."
if ! command -v uv >/dev/null 2>&1; then
  echo "'uv' not found. Attempting to install via pip..."
  python3 -m pip install --upgrade uv
fi

echo "[2/4] Creating/using virtual environment (.venv) with uv..."
if [[ ! -d .venv ]]; then
  uv venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[3/4] Installing requirements with uv pip..."
if [[ -f requirements.txt ]]; then
  uv pip install -r requirements.txt
else
  echo "requirements.txt not found in $PROJECT_ROOT" >&2
  exit 1
fi

# Optional: expose device override (cpu|cuda|mps) by env var DEVICE
DEVICE_ARG=${DEVICE:-}
if [[ -n "${DEVICE_ARG}" ]]; then
  DEVICE_FLAG=(--device "${DEVICE_ARG}")
else
  DEVICE_FLAG=()
fi

echo "[4/4] Running training scripts (comment out lines you don't need)"

# --- Stage 1: Masked sensor imputation ---
# Uncomment/comment different strategies to experiment with different approaches

# Strategy 1: Balanced MSE with per-task losses (recommended starting point)
python train_masked.py \
  --epochs 15 \
  --batch_size 64 \
  --mask_ratio 0.6 \
  --loss_type balanced_mse \
  --thm_loss_type mse \
  --tof_loss_type huber \
  --huber_beta 1.0 \
  "${DEVICE_FLAG[@]}"

# Strategy 2: With mask-aware conditioning (includes masked THM/TOF in encoder)
# python train_masked.py \
#   --epochs 15 \
#   --batch_size 64 \
#   --mask_ratio 0.6 \
#   --loss_type balanced_mse \
#   --thm_loss_type mse \
#   --tof_loss_type huber \
#   --huber_beta 1.0 \
#   --use_mask_conditioning \
#   "${DEVICE_FLAG[@]}"

# Strategy 3: Shared decoder trunk (encourages cross-modal representations)
# python train_masked.py \
#   --epochs 15 \
#   --batch_size 64 \
#   --mask_ratio 0.6 \
#   --loss_type balanced_mse \
#   --thm_loss_type mse \
#   --tof_loss_type huber \
#   --huber_beta 1.0 \
#   --use_shared_decoder \
#   "${DEVICE_FLAG[@]}"

# Strategy 4: Adaptive weighted (learnable task weights)
# python train_masked.py \
#   --epochs 15 \
#   --batch_size 64 \
#   --mask_ratio 0.6 \
#   --loss_type adaptive_weighted \
#   --thm_loss_type mse \
#   --tof_loss_type huber \
#   --huber_beta 1.0 \
#   "${DEVICE_FLAG[@]}"

# Strategy 5: Higher mask ratio for stronger denoising
# python train_masked.py \
#   --epochs 20 \
#   --batch_size 64 \
#   --mask_ratio 0.7 \
#   --loss_type balanced_mse \
#   --thm_loss_type huber \
#   --tof_loss_type huber \
#   --huber_beta 1.0 \
#   "${DEVICE_FLAG[@]}"

# Strategy 6: Conservative approach (lower mask ratio, robust losses)
# python train_masked.py \
#   --epochs 12 \
#   --batch_size 64 \
#   --mask_ratio 0.4 \
#   --loss_type balanced_mse \
#   --thm_loss_type huber \
#   --tof_loss_type huber \
#   --huber_beta 0.5 \
#   "${DEVICE_FLAG[@]}"

# Strategy 7: Kitchen sink (all features enabled)
# python train_masked.py \
#   --epochs 20 \
#   --batch_size 64 \
#   --mask_ratio 0.6 \
#   --loss_type balanced_mse \
#   --thm_loss_type mse \
#   --tof_loss_type huber \
#   --huber_beta 1.0 \
#   --use_mask_conditioning \
#   --use_shared_decoder \
#   "${DEVICE_FLAG[@]}"

# --- Stage 2: Binary classification ---
# Uncomment to run
# python train_binary.py \
#   --epochs 3 \
#   --batch_size 64 \
#   --lr 1e-3 \
#   "${DEVICE_FLAG[@]}"

# --- Stage 3: Multimodal gesture classification ---
# Uncomment to run
# python train_multimodal.py \
#   --epochs 10 \
#   --batch_size 32 \
#   --lr 1e-4 \
#   "${DEVICE_FLAG[@]}"

echo "Done."