#!/usr/bin/env python3
import argparse
import os
from PIL import Image, ImageSequence

def process_one(input_dir: str, output_dir: str, name: str, method: str, expected_frames: int = 193, number: int = 0):
    gif_path = os.path.join(input_dir, name, method, str(number), f"{number}_video.gif")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(gif_path):
        print(f"[WARN] Missing: {gif_path}")
        return

    try:
        im = Image.open(gif_path)
    except Exception as e:
        print(f"[ERROR] Failed to open {gif_path}: {e}")
        return

    count = 0
    for i, frame in enumerate(ImageSequence.Iterator(im)):
        # Convert to RGB to avoid palette/LA modes; then take red channel.
        red = frame.convert("RGB").getchannel("R")  # single-channel (L)
        out_name = f"{name}_{method}_0_0_{i:03d}.tiff"
        out_path = os.path.join(output_dir, out_name)
        try:
            red.save(out_path, format="TIFF", compression="tiff_deflate")
        except Exception as e:
            print(f"[ERROR] Saving {out_path} failed: {e}")
            continue
        count += 1

    if count != expected_frames:
        print(f"[NOTE] {gif_path}: saved {count} frames (expected {expected_frames}).")
    else:
        print(f"[OK] {gif_path}: saved {count} frames.")

def read_list(cli_list, file_path):
    items = list(cli_list) if cli_list else []
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            items.extend([line.strip() for line in f if line.strip()])
    return items

def main():
    ap = argparse.ArgumentParser(description="Extract red channel from GIF frames to TIFFs.")
    ap.add_argument("--input-dir", required=True, help="Root directory containing {name}/{method}/{number}/{number}_video.gif")
    ap.add_argument("--output-dir", required=True, help="Directory for output TIFF files")
    ap.add_argument("--number", type=int, default=0, help="")
    ap.add_argument("--names", nargs="*", default=[], help="List of names")
    ap.add_argument("--methods", nargs="*", default=[], help="List of methods")
    ap.add_argument("--names-file", type=str, default=None, help="Optional file with one name per line")
    ap.add_argument("--methods-file", type=str, default=None, help="Optional file with one method per line")
    ap.add_argument("--expected-frames", type=int, default=193, help="Expected frames per GIF (default 193)")
    args = ap.parse_args()

    names = read_list(args.names, args.names_file)
    methods = read_list(args.methods, args.methods_file)

    if not names or not methods:
        raise SystemExit("Please provide at least one --names and one --methods (or their *_file variants).")

    for name in names:
        for method in methods:
            process_one(args.input_dir, args.output_dir, name, method, args.expected_frames, args.number)

if __name__ == "__main__":
    main()
