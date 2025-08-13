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
VENV_CREATED=0
if [[ ! -d .venv ]]; then
  uv venv .venv
  VENV_CREATED=1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Decide whether to install dependencies
INSTALL_DEPS=0
if [[ "${FORCE_INSTALL:-0}" == "1" ]]; then
  INSTALL_DEPS=1
elif [[ ${VENV_CREATED} -eq 1 ]]; then
  INSTALL_DEPS=1
fi

if [[ ${INSTALL_DEPS} -eq 1 ]]; then
  echo "[3/4] Installing requirements with uv pip..."
  if [[ -f requirements.txt ]]; then
    uv pip install -r requirements.txt
  else
    echo "requirements.txt not found in $PROJECT_ROOT" >&2
    exit 1
  fi
else
  echo "[3/4] Skipping requirements install (existing .venv). Set FORCE_INSTALL=1 to reinstall."
fi

# Optional: expose device override (cpu|cuda|mps) by env var DEVICE
DEVICE_ARG=${DEVICE:-}
# Pre-declare array to prevent "unbound variable" errors under `set -u`
declare -a DEVICE_FLAG=()
if [[ -n "${DEVICE_ARG}" ]]; then
  DEVICE_FLAG=(--device "${DEVICE_ARG}")
fi

# Optional: expose optimizer and scheduler overrides
OPTIMIZER_ARG=${OPTIMIZER:-adam}
SCHEDULER_ARG=${SCHEDULER:-cosine}

# Combine all common flags
ALL_FLAGS=(--optimizer "${OPTIMIZER_ARG}" --scheduler "${SCHEDULER_ARG}" "${DEVICE_FLAG[@]+"${DEVICE_FLAG[@]}"}")

echo "[4/4] Running training scripts (comment out lines you don't need)"

# --- Stage 1: Masked sensor imputation ---
# Uncomment/comment different strategies to experiment with different approaches

# # Strategy 1: Balanced MSE with per-task losses (recommended starting point)
# python train_masked.py \
#   --epochs 5 \
#   --batch_size 64 \
#   --mask_ratio 0.2 \
#   --loss_type conf_mse \
#   # --thm_loss_type mse \
#   # --tof_loss_type huber \
#   --huber_beta 1.0 \
#   "${ALL_FLAGS[@]}"

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
#   "${ALL_FLAGS[@]}"

# # Strategy 3: UNet-style upsampling trunk (shared feature maps)
# python train_masked.py \
#   --epochs 5 \
#   --batch_size 128 \
#   --mask_ratio 0.3 \
#   --imu_mask_ratio 0.2 \
#   --thm_loss_type mse \
#   --tof_loss_type huber \
#   --imu_loss_type mae \
#   --huber_beta 1.0 \
#   --use_unet_decoder \
#   --use_mask_conditioning \
#   --use_task_attention \
#   "${ALL_FLAGS[@]}"

# Strategy 4: Adaptive weighted (learnable task weights)
# python train_masked.py \
#   --epochs 15 \
#   --batch_size 64 \
#   --mask_ratio 0.6 \
#   --loss_type adaptive_weighted \
#   --thm_loss_type mse \
#   --tof_loss_type huber \
#   --huber_beta 1.0 \
#   "${ALL_FLAGS[@]}"

# Strategy 5: Higher mask ratio for stronger denoising
# python train_masked.py \
#   --epochs 20 \
#   --batch_size 64 \
#   --mask_ratio 0.7 \
#   --loss_type balanced_mse \
#   --thm_loss_type huber \
#   --tof_loss_type huber \
#   --huber_beta 1.0 \
#   "${ALL_FLAGS[@]}"

# Strategy 6: Conservative approach (lower mask ratio, robust losses)
# python train_masked.py \
#   --epochs 12 \
#   --batch_size 64 \
#   --mask_ratio 0.4 \
#   --loss_type balanced_mse \
#   --thm_loss_type huber \
#   --tof_loss_type huber \
#   --huber_beta 0.5 \
#   "${ALL_FLAGS[@]}"

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
#   "${ALL_FLAGS[@]}"

# --- Stage 2: Binary classification ---
# Uncomment to run
# python train_binary.py \
#   --epochs 3 \
#   --batch_size 64 \
#   --lr 1e-3 \
#   "${ALL_FLAGS[@]}"

# --- Stage 3: Multimodal gesture classification ---
# Uncomment to run
python train_multimodal.py \
  --data_dir 'preprocessed' \
  --epochs 2 \
  --batch_size 128 \
  --lr 3e-4 \
  --tof_img_size 64 \
  --num_workers 6 \
  --task all \
  # --no_freeze_backbone \
  "${ALL_FLAGS[@]}"

# # --- IMU 2D spectrogram classification (IMU-only) ---
# python train_imu2d.py \
#   --data_dir 'preprocessed' \
#   --epochs 2 \
#   --batch_size 128 \
#   --img_hw 128x64 \
#   --lr 1e-3 \
#   --transform stft \
#   --num_workers 6 \
#   # --no_freeze_backbone \
#   "${ALL_FLAGS[@]}"

echo "Done."