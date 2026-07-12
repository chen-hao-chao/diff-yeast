#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="/home/chchao0/projects/aip-rahulgk/chchao0/generated_video"
NAMES="YKL007W"
METHODS="nucleus"
OUT_PREFIX="${NAMES}/extracted_green_${NAMES}_"

for i in {0..9}; do
  echo ">>> Running number=$i"
  python extract_channels_green.py \
    --input-dir "$INPUT_DIR" \
    --names "$NAMES" \
    --methods "$METHODS" \
    --output-dir "${OUT_PREFIX}${i}" \
    --number "$i"
done