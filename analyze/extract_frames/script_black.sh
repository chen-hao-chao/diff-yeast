#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="/home/chchao0/projects/aip-rahulgk/chchao0/generated_video"
NAMES="YNL118C" # YDR458C, YKR083C, YNL118C
METHODS="nucleus"
OUT_PREFIX="${NAMES}/extracted_black_${NAMES}_"

for i in {0..9}; do
  echo ">>> Running number=$i"
  python extract_channels_black.py \
    --input-dir "$INPUT_DIR" \
    --names "$NAMES" \
    --threshold 11 \
    --methods "$METHODS" \
    --output-dir "${OUT_PREFIX}${i}" \
    --number "$i"
done