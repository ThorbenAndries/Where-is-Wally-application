#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/training"
LOG_FILE="$LOG_DIR/train-$TIMESTAMP.log"
mkdir -p "$LOG_DIR" "$ROOT_DIR/models"

exec > >(tee -a "$LOG_FILE") 2>&1

TRAIN_IMAGES="${TRAIN_IMAGES:-3000}"
VAL_IMAGES="${VAL_IMAGES:-500}"
SYNTH_EPOCHS="${SYNTH_EPOCHS:-100}"
CVAT_EPOCHS="${CVAT_EPOCHS:-40}"
IMAGE_SIZE="${IMAGE_SIZE:-640}"
DEVICE="${DEVICE:-0}"

echo "Training started: $(date)"
echo "Log file: $LOG_FILE"
echo "Synthetic images: train=$TRAIN_IMAGES val=$VAL_IMAGES"
echo "Epochs: synthetic=$SYNTH_EPOCHS cvat=$CVAT_EPOCHS"
echo "Image size: $IMAGE_SIZE"
echo "Device: $DEVICE"

source "$ROOT_DIR/venv/bin/activate"

python tools/generate_synthetic_mascot_dataset.py \
  --output data/my-mascot \
  --train "$TRAIN_IMAGES" \
  --val "$VAL_IMAGES" \
  --difficulty hard \
  --mascot assets/mascot.png \
  --backgrounds assets/backgrounds

yolo detect train \
  data=data/my-mascot/data.yaml \
  model=yolov8n.pt \
  epochs="$SYNTH_EPOCHS" \
  imgsz="$IMAGE_SIZE" \
  patience=25 \
  device="$DEVICE" \
  project=runs/detect \
  name=mascot-synthetic \
  exist_ok=True

SYNTHETIC_MODEL="$(find "$ROOT_DIR/runs" -path '*/mascot-synthetic/weights/best.pt' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -z "$SYNTHETIC_MODEL" ]]; then
  echo "Could not find mascot-synthetic best.pt"
  exit 1
fi
echo "Synthetic model: $SYNTHETIC_MODEL"

yolo detect train \
  data=data/cvat-mascot/data.yaml \
  model="$SYNTHETIC_MODEL" \
  epochs="$CVAT_EPOCHS" \
  imgsz="$IMAGE_SIZE" \
  patience=12 \
  lr0=0.001 \
  device="$DEVICE" \
  project=runs/detect \
  name=mascot-final \
  exist_ok=True

FINAL_MODEL="$(find "$ROOT_DIR/runs" -path '*/mascot-final/weights/best.pt' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -z "$FINAL_MODEL" ]]; then
  echo "Could not find mascot-final best.pt"
  exit 1
fi
cp "$FINAL_MODEL" models/mascot.pt

echo "Training finished: $(date)"
echo "Final model: $ROOT_DIR/models/mascot.pt"
echo "Synthetic model: $SYNTHETIC_MODEL"
echo "Fine-tuned model: $FINAL_MODEL"
