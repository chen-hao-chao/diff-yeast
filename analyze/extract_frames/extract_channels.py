#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image, ImageSequence
import numpy as np

REAL_IDXS = {0, 64, 96, 128, 160, 192}


def extract_channels(frame_rgb: Image.Image):
    """Return (green_L, red_L, shape_mask_L) as PIL L-mode images."""
    arr = np.array(frame_rgb, dtype=np.uint8)
    r = arr[..., 0]
    g = arr[..., 1]
    mask = (arr.any(axis=-1)).astype(np.uint8) * 255  # non-black mask

    red_L = Image.fromarray(r, mode='L')
    green_L = Image.fromarray(g, mode='L')
    shape_L = Image.fromarray(mask, mode='L')
    return green_L, red_L, shape_L


def save_frames_as_images(frames, out_dir: Path, base_name: str, ext: str):
    """Save each frame as an individual image file (TIFF or PNG)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(out_dir / f"{base_name}_frame{i:03d}.{ext}")


def process(prefix: str, name: str, method: str, out_dir: Path):
    gif_path = Path(prefix) / name / method / "0" / "0_video.gif"
    if not gif_path.exists():
        raise FileNotFoundError(f"GIF not found: {gif_path}")

    # Collect frames by class
    real_green, real_red, real_shape = [], [], []
    fake_green, fake_red, fake_shape = [], [], []

    with Image.open(gif_path) as im:
        for idx, frame in enumerate(ImageSequence.Iterator(im)):
            rgb = frame.convert("RGB")
            gL, rL, sL = extract_channels(rgb)
            if idx in REAL_IDXS:
                real_green.append(gL)
                real_red.append(rL)
                real_shape.append(sL)
            else:
                fake_green.append(gL)
                fake_red.append(rL)
                fake_shape.append(sL)

    base_name = f"{name}_{method}_0_0"

    # === Real ===
    save_frames_as_images(real_green, out_dir / "real" / "green" / "tiff", base_name, "tiff")
    save_frames_as_images(real_red,   out_dir / "real" / "red"   / "tiff", base_name, "tiff")
    save_frames_as_images(real_shape, out_dir / "real" / "shape" / "tiff", base_name, "tiff")

    save_frames_as_images(real_green, out_dir / "real" / "green" / "png", base_name, "png")
    save_frames_as_images(real_red,   out_dir / "real" / "red"   / "png", base_name, "png")
    save_frames_as_images(real_shape, out_dir / "real" / "shape" / "png", base_name, "png")

    # === Fake ===
    save_frames_as_images(fake_green, out_dir / "fake" / "green" / "tiff", base_name, "tiff")
    save_frames_as_images(fake_red,   out_dir / "fake" / "red"   / "tiff", base_name, "tiff")
    save_frames_as_images(fake_shape, out_dir / "fake" / "shape" / "tiff", base_name, "tiff")

    save_frames_as_images(fake_green, out_dir / "fake" / "green" / "png", base_name, "png")
    save_frames_as_images(fake_red,   out_dir / "fake" / "red"   / "png", base_name, "png")
    save_frames_as_images(fake_shape, out_dir / "fake" / "shape" / "png", base_name, "png")

    print("Saved TIFF and PNG images under:", out_dir)


def main():
    p = argparse.ArgumentParser(description="Extract red/green/shape TIFF+PNG per-frame images from GIF.")
    p.add_argument("--prefix", default="./")
    p.add_argument("--name", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--out", default="out")
    args = p.parse_args()

    process(args.prefix, args.name, args.method, Path(args.out))


if __name__ == "__main__":
    main()
