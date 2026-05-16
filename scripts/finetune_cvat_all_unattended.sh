#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/training"
LOG_FILE="$LOG_DIR/finetune-cvat-all-$TIMESTAMP.log"
mkdir -p "$LOG_DIR" "$ROOT_DIR/models"

exec > >(tee -a "$LOG_FILE") 2>&1

BASE_MODEL="${BASE_MODEL:-models/mascot.pt}"
EPOCHS="${EPOCHS:-140}"
IMAGE_SIZE="${IMAGE_SIZE:-1280}"
DEVICE="${DEVICE:-0}"
RUN_NAME="${RUN_NAME:-mascot-cvat-all-v2}"

echo "CVAT fine-tune started: $(date)"
echo "Log file: $LOG_FILE"
echo "Base model: $BASE_MODEL"
echo "Dataset: data/cvat-all-mascot-clean/data.yaml"
echo "Epochs: $EPOCHS"
echo "Image size: $IMAGE_SIZE"
echo "Device: $DEVICE"

source "$ROOT_DIR/venv/bin/activate"

yolo detect train \
  data=data/cvat-all-mascot-clean/data.yaml \
  model="$BASE_MODEL" \
  epochs="$EPOCHS" \
  imgsz="$IMAGE_SIZE" \
  patience=35 \
  lr0=0.0005 \
  batch=8 \
  device="$DEVICE" \
  project=runs/detect \
  name="$RUN_NAME" \
  exist_ok=True

BEST_MODEL="$(find "$ROOT_DIR/runs" -path "*/$RUN_NAME/weights/best.pt" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -z "$BEST_MODEL" ]]; then
  echo "Could not find best model for $RUN_NAME"
  exit 1
fi

cp "$BEST_MODEL" models/mascot.pt

echo "CVAT fine-tune finished: $(date)"
echo "Best model: $BEST_MODEL"
echo "Updated app model: $ROOT_DIR/models/mascot.pt"
