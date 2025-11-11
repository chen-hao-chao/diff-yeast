#!/usr/bin/env python3
import argparse
import os
from PIL import Image, ImageSequence
import numpy as np

def mk_nonblack_mask(frame: Image.Image, threshold: int = 0, use_alpha: bool = False) -> Image.Image:
    """Return an 8-bit mask (L) where pixels > threshold in any RGB channel are 255."""
    rgba = frame.convert("RGBA")
    arr = np.array(rgba, dtype=np.uint8)
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    nonblack = (r > threshold) | (g > threshold) | (b > threshold)
    if use_alpha:
        nonblack &= (a > 0)
    return Image.fromarray(np.where(nonblack, 255, 0).astype(np.uint8), mode="L")

def process_one(input_dir, output_dir, name, method, expected_frames=193, threshold=0, use_alpha=False, number: int = 0):
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
        mask = mk_nonblack_mask(frame, threshold=threshold, use_alpha=use_alpha)
        out_name = f"{name}_{method}_0_0_{i:03d}.tiff"   # <── padded index
        out_path = os.path.join(output_dir, out_name)
        try:
            mask.save(out_path, format="TIFF", compression="tiff_deflate")
        except Exception as e:
            print(f"[ERROR] Saving {out_path} failed: {e}")
            continue
        count += 1

    msg = "[OK]" if count == expected_frames else "[NOTE]"
    print(f"{msg} {gif_path}: saved {count} frames (expected {expected_frames}).")

def read_list(cli_list, file_path):
    items = list(cli_list) if cli_list else []
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            items.extend([line.strip() for line in f if line.strip()])
    return items

def main():
    ap = argparse.ArgumentParser(description="Create non-black segmentation masks from GIF frames.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--number", type=int, default=0, help="")
    ap.add_argument("--names", nargs="*", default=[])
    ap.add_argument("--methods", nargs="*", default=[])
    ap.add_argument("--names-file", type=str, default=None)
    ap.add_argument("--methods-file", type=str, default=None)
    ap.add_argument("--expected-frames", type=int, default=193)
    ap.add_argument("--threshold", type=int, default=0)
    ap.add_argument("--use-alpha", action="store_true")
    args = ap.parse_args()

    names = read_list(args.names, args.names_file)
    methods = read_list(args.methods, args.methods_file)
    if not names or not methods:
        raise SystemExit("Please provide at least one --names and one --methods (or *_file variants).")

    for name in names:
        for method in methods:
            process_one(args.input_dir, args.output_dir, name, method,
                        expected_frames=args.expected_frames,
                        threshold=args.threshold, use_alpha=args.use_alpha, number=args.number)

if __name__ == "__main__":
    main()
