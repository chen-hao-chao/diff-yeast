import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence

def get_mask_area(frame: Image.Image, threshold: int = 0, use_alpha: bool = False) -> int:
    """Calculates the area (count of non-black pixels) in a single frame."""
    rgba = frame.convert("RGBA")
    arr = np.array(rgba)
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    
    # Define non-black pixels
    nonblack = (r > threshold) | (g > threshold) | (b > threshold)
    if use_alpha:
        nonblack &= (a > 0)
    
    # Area is the total number of True values in the mask
    return np.sum(nonblack)

def process_and_plot(input_dir, name, method, threshold=0, use_alpha=False):
    """Processes 10 clips and generates a 5x2 plot of areas over time."""
    fig, axes = plt.subplots(5, 2, figsize=(12, 15), constrained_layout=True)
    axes = axes.flatten() # Flatten to 1D for easy indexing (0-9)

    # Assuming clips are numbered 0 to 9 based on your requirement
    for clip_idx in range(10):
        gif_path = os.path.join(input_dir, name, method, str(clip_idx), f"{clip_idx}_video.gif")
        
        areas = []
        if not os.path.isfile(gif_path):
            print(f"[WARN] Missing: {gif_path}")
            axes[clip_idx].set_title(f"Clip {clip_idx} (Missing)")
            continue

        try:
            im = Image.open(gif_path)
            for frame in ImageSequence.Iterator(im):
                area = get_mask_area(frame, threshold, use_alpha)
                areas.append(area)
            
            # Plotting on the specific subplot
            ax = axes[clip_idx]
            ax.plot(areas, color='steelblue')
            ax.set_title(f"Clip {clip_idx}")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Area")
            ax.set_ylim(700, 2400)
            ax.grid(True, linestyle='--', alpha=0.6)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {gif_path}: {e}")
            axes[clip_idx].set_title(f"Clip {clip_idx} (Error)")

    plt.suptitle(f"Segmentation Area Over Time: {name} - {method}", fontsize=16)
    plt.savefig(f'extract_and_plot_{name}_{method}.png')

def main():
    ap = argparse.ArgumentParser(description="Plot non-black segmentation areas from GIF frames.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--threshold", type=int, default=0)
    ap.add_argument("--use-alpha", action="store_true")
    args = ap.parse_args()

    for name in args.names:
        for method in args.methods:
            process_and_plot(args.input_dir, name, method, 
                             threshold=args.threshold, use_alpha=args.use_alpha)

if __name__ == "__main__":
    main()