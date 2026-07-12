#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="/home/chchao0/projects/aip-rahulgk/chchao0/generated_video"
NAMES="YKL007W"
METHODS="nucleus"

python extract_and_plot_area.py \
  --input-dir "$INPUT_DIR" \
  --names "$NAMES" \
  --threshold 11 \
  --methods "$METHODS"