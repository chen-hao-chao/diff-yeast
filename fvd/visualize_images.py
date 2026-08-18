"""
Visualize the Opera Phenix plate images in /datasets/yeast-imgs/gt_video/videos/Images.

Filename pattern: r{row}c{col}f{field}p{plane}-ch{channel}sk{frame}fk1fl1.tiff
  row/col : well position on the plate (see PROTEIN_TABLE below)
  field   : field of view within the well (f01-f24)
  channel : ch1 phase contrast, ch2 protein-GFP, ch3 nucleus+septin+cytoplasm
            bleed-through, ch4 cytoplasm (TDH3pr-E2Crimson)
  frame   : sk1-sk20, timepoints of the video

Visualizations supported:
  montage  - one well/field/timepoint, all 4 channels + an RGB composite, saved as PNG
  video    - one well/field across all timepoints, saved as GIF and/or MP4
  plate    - one composite thumbnail per well (single field/timepoint), saved as PNG
  segment  - threshold ch1 (phase contrast) to find cells, saved as PNG
  list     - print the wells that actually have image data, with protein names
"""

import argparse
import re
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

IMAGES_DIR = Path("/datasets/yeast-imgs/gt_video/videos/Images")
OUT_DIR = Path(__file__).resolve().parent / "viz_out"

ROW_LETTERS = "ABCDEFGHIJKLMNOP"

# (row letter, column) -> (protein name, ORF)
PROTEIN_TABLE = {
    ("A", 4): ("Nuf2", "YOL069W"),
    ("A", 6): ("Pre2", "YPR103W"),
    ("A", 8): ("Sla2", "YNL243W"),
    ("A", 10): ("Nhp6a", "YPR052C"),
    ("A", 12): ("Snf7", "YLR025W"),
    ("A", 14): ("Dad2", "YKR083C"),
    ("A", 16): ("Spf1", "YEL031W"),
    ("A", 18): ("Dcp2", "YNL118C"),
    ("A", 20): ("Mca1", "YOR197W"),
    ("A", 22): ("Sec7", "YDR170C"),
    ("A", 24): ("Psr1", "YLL010C"),
    ("B", 2): ("Nop10", "YHR072W-A"),
    ("B", 4): ("Mid2", "YLR332W"),
    ("B", 6): ("Pil1", "YGR086C"),
    ("B", 10): ("Rad52", "YML032C"),
    ("B", 12): ("Sec21", "YNL287W"),
    ("B", 14): ("Vph1", "YOR270C"),
    ("B", 16): ("Atg8", "YBL078C"),
    ("B", 18): ("Om45", "YIL136W"),
    ("B", 20): ("Nuc1", "YJL208C"),
    ("B", 22): ("HTA2", "YBL003C"),
}
NAME_TO_WELL = {name.lower(): rc for rc, (name, _orf) in PROTEIN_TABLE.items()}

CHANNEL_LABELS = {
    1: "ch1 phase contrast",
    2: "ch2 protein-GFP",
    3: "ch3 nucleus+septin+cytoplasm bleed",
    4: "ch4 cytoplasm (E2Crimson)",
}

FRAME_RE = re.compile(r"r(\d{2})c(\d{2})f(\d{2})p\d{2}-ch(\d)sk(\d+)fk1fl1\.tiff$")


def row_letter_to_num(letter: str) -> int:
    return ROW_LETTERS.index(letter.upper()) + 1


def row_num_to_letter(num: int) -> str:
    return ROW_LETTERS[num - 1]


def frame_path(row: int, col: int, field: int, channel: int, sk: int) -> Path:
    return IMAGES_DIR / f"r{row:02d}c{col:02d}f{field:02d}p01-ch{channel}sk{sk}fk1fl1.tiff"


def load_frame(row: int, col: int, field: int, channel: int, sk: int) -> np.ndarray:
    path = frame_path(row, col, field, channel, sk)
    if not path.exists():
        raise FileNotFoundError(path)
    return tifffile.imread(path)


def discover_wells() -> list[tuple[int, int]]:
    """Return sorted (row, col) pairs that actually have image files on disk."""
    seen = set()
    for p in IMAGES_DIR.iterdir():
        m = FRAME_RE.match(p.name)
        if m:
            seen.add((int(m.group(1)), int(m.group(2))))
    return sorted(seen)


def resolve_well(args) -> tuple[int, int]:
    if args.protein:
        key = args.protein.lower()
        if key not in NAME_TO_WELL:
            available = ", ".join(sorted(n for n, _ in PROTEIN_TABLE.values()))
            raise SystemExit(f"Unknown protein '{args.protein}'. Available: {available}")
        letter, col = NAME_TO_WELL[key]
        return row_letter_to_num(letter), col
    if args.row is None or args.col is None:
        raise SystemExit("Provide --protein NAME, or both --row and --col")
    return row_letter_to_num(args.row), args.col


def to_uint8(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1
    scaled = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8)


COMPOSITE_MODES = {
    "rgb": "R=ch3 G=ch2 B=ch4",
    "clean": "R=ch3-ch4 (cytoplasm removed) G=ch2",
    "no_ch4": "R=ch3 G=ch2 (ch4 ignored)",
}


def composite_rgb(row: int, col: int, field: int, sk: int, mode: str = "clean") -> np.ndarray:
    g = to_uint8(load_frame(row, col, field, 2, sk))

    if mode == "rgb":
        r = to_uint8(load_frame(row, col, field, 3, sk))
        b = to_uint8(load_frame(row, col, field, 4, sk))
        return np.stack([r, g, b], axis=-1)

    if mode == "clean":
        ch3 = load_frame(row, col, field, 3, sk).astype(np.float32)
        ch4 = load_frame(row, col, field, 4, sk).astype(np.float32)
        diff = np.clip(ch3 - ch4, 0, None)
        r = to_uint8(diff)
        b = np.zeros_like(r)
        return np.stack([r, g, b], axis=-1)

    if mode == "no_ch4":
        r = to_uint8(load_frame(row, col, field, 3, sk))
        b = np.zeros_like(r)
        return np.stack([r, g, b], axis=-1)

    raise ValueError(f"Unknown composite mode: {mode}")


def segment_ch1(
    row: int,
    col: int,
    field: int,
    sk: int,
    threshold: int | None = None,
    min_size: int = 15,
    open_radius: int = 1,
) -> dict:
    """Threshold ch1 (phase contrast) to find cells.

    threshold: 0-255 value on the contrast-stretched image; None -> Otsu.
    min_size: drop connected components smaller than this many pixels.
    open_radius: morphological opening kernel radius, removes speckle noise.
    """
    img8 = to_uint8(load_frame(row, col, field, 1, sk))
    blurred = cv2.GaussianBlur(img8, (3, 3), 0)

    if threshold is None:
        used_threshold, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        used_threshold, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    if open_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_radius + 1, 2 * open_radius + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [i for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= min_size]
    clean_mask = np.isin(labels, keep).astype(np.uint8) * 255

    overlay = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

    return {
        "img8": img8,
        "mask": clean_mask,
        "overlay": overlay,
        "count": len(keep),
        "threshold": used_threshold,
        "areas": [stats[i, cv2.CC_STAT_AREA] for i in keep],
    }


def well_label(row: int, col: int) -> str:
    letter = row_num_to_letter(row)
    hit = PROTEIN_TABLE.get((letter, col))
    if hit:
        name, orf = hit
        return f"{letter}{col} {name} ({orf})"
    return f"{letter}{col}"


def cmd_list(_args):
    wells = discover_wells()
    print(f"{len(wells)} wells found under {IMAGES_DIR}\n")
    for row, col in wells:
        print(" ", well_label(row, col))


def cmd_montage(args):
    row, col = resolve_well(args)
    field, sk = args.field, args.sk

    mask_bool = None
    title_suffix = ""
    if args.mask:
        seg = segment_ch1(
            row, col, field, sk,
            threshold=args.threshold,
            min_size=args.min_size,
            open_radius=args.open_radius,
        )
        mask_bool = seg["mask"] > 0
        title_suffix = f"  |  {seg['count']} cells (mask threshold={seg['threshold']:.0f})"

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.3))
    for ax, ch in zip(axes[:4], (1, 2, 3, 4)):
        img = to_uint8(load_frame(row, col, field, ch, sk))
        if mask_bool is not None:
            img = img.copy()
            img[~mask_bool] = 0
        ax.imshow(img, cmap="gray")
        ax.set_title(CHANNEL_LABELS[ch], fontsize=9)
        ax.axis("off")

    composite = composite_rgb(row, col, field, sk, mode=args.mode)
    if mask_bool is not None:
        composite = composite.copy()
        composite[~mask_bool] = 0
    axes[4].imshow(composite)
    axes[4].set_title(f"composite {COMPOSITE_MODES[args.mode]}", fontsize=9)
    axes[4].axis("off")

    fig.suptitle(f"{well_label(row, col)}  field {field:02d}  frame sk{sk}{title_suffix}")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or OUT_DIR / f"montage_r{row:02d}c{col:02d}_f{field:02d}_sk{sk}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def cmd_segment(args):
    row, col = resolve_well(args)
    field, sk = args.field, args.sk

    result = segment_ch1(
        row, col, field, sk,
        threshold=args.threshold,
        min_size=args.min_size,
        open_radius=args.open_radius,
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(result["img8"], cmap="gray")
    axes[0].set_title("ch1 phase contrast", fontsize=9)
    axes[1].imshow(result["mask"], cmap="gray")
    axes[1].set_title(f"mask (threshold={result['threshold']:.0f})", fontsize=9)
    axes[2].imshow(result["overlay"])
    axes[2].set_title(f"overlay: {result['count']} cells", fontsize=9)
    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"{well_label(row, col)}  field {field:02d}  frame sk{sk}")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or OUT_DIR / f"segment_r{row:02d}c{col:02d}_f{field:02d}_sk{sk}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"threshold={result['threshold']:.1f}  cells={result['count']}  median area={np.median(result['areas']) if result['areas'] else 0:.0f}px")


def cmd_video(args):
    row, col = resolve_well(args)
    field = args.field
    sk_range = range(args.sk_start, args.sk_end + 1)

    frames = []
    for sk in sk_range:
        frame = composite_rgb(row, col, field, sk, mode=args.mode)
        if args.mask:
            seg = segment_ch1(
                row, col, field, sk,
                threshold=args.threshold,
                min_size=args.min_size,
                open_radius=args.open_radius,
            )
            frame = frame.copy()
            frame[seg["mask"] == 0] = 0
        frames.append(frame)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"video_r{row:02d}c{col:02d}_f{field:02d}"

    if args.gif:
        gif_path = args.out or OUT_DIR / f"{stem}.gif"
        imageio.mimsave(gif_path, frames, fps=args.fps)
        print(f"Saved {gif_path}")

    if args.mp4:
        mp4_path = OUT_DIR / f"{stem}.mp4" if args.gif or not args.out else args.out
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4_path), fourcc, args.fps, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Saved {mp4_path}")


def cmd_plate(args):
    wells = discover_wells()
    ncols = 11
    nrows = -(-len(wells) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows))
    axes = np.atleast_2d(axes)

    for idx, (row, col) in enumerate(wells):
        ax = axes[idx // ncols, idx % ncols]
        img = composite_rgb(row, col, args.field, args.sk, mode=args.mode)
        ax.imshow(img)
        ax.set_title(well_label(row, col), fontsize=8)
        ax.axis("off")

    for idx in range(len(wells), nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    fig.suptitle(f"Plate overview  field {args.field:02d}  frame sk{args.sk}")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or OUT_DIR / f"plate_f{args.field:02d}_sk{args.sk}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_well_args(p):
        p.add_argument("--protein", help="protein name, e.g. Nuf2 (case-insensitive)")
        p.add_argument("--row", help="plate row letter, e.g. A")
        p.add_argument("--col", type=int, help="plate column, e.g. 4")

    p_list = sub.add_parser("list", help="list wells with data + protein names")
    p_list.set_defaults(func=cmd_list)

    def add_mode_arg(p):
        p.add_argument(
            "--mode",
            choices=sorted(COMPOSITE_MODES),
            default="clean",
            help="composite color mapping: 'rgb' = R=ch3 G=ch2 B=ch4, "
            "'clean' = R=(ch3-ch4) cytoplasm-subtracted G=ch2 (default)",
        )

    p_montage = sub.add_parser("montage", help="all 4 channels + composite for one well/field/frame")
    add_well_args(p_montage)
    p_montage.add_argument("--field", type=int, default=1)
    p_montage.add_argument("--sk", type=int, default=1)
    add_mode_arg(p_montage)
    p_montage.add_argument("--mask", action="store_true", help="zero out background using a ch1 threshold mask")
    p_montage.add_argument("--threshold", type=int, default=None, help="0-255 fixed threshold for --mask; default: Otsu (auto)")
    p_montage.add_argument("--min-size", type=int, default=15, help="drop mask components smaller than this many pixels")
    p_montage.add_argument("--open-radius", type=int, default=1, help="mask morphological opening kernel radius")
    p_montage.add_argument("--out", type=Path)
    p_montage.set_defaults(func=cmd_montage)

    p_segment = sub.add_parser("segment", help="threshold ch1 to segment cells")
    add_well_args(p_segment)
    p_segment.add_argument("--field", type=int, default=1)
    p_segment.add_argument("--sk", type=int, default=1)
    p_segment.add_argument("--threshold", type=int, default=None, help="0-255 fixed threshold; default: Otsu (auto)")
    p_segment.add_argument("--min-size", type=int, default=15, help="drop components smaller than this many pixels")
    p_segment.add_argument("--open-radius", type=int, default=1, help="morphological opening kernel radius")
    p_segment.add_argument("--out", type=Path)
    p_segment.set_defaults(func=cmd_segment)

    p_video = sub.add_parser("video", help="composite video across all timepoints for one well/field")
    add_well_args(p_video)
    p_video.add_argument("--field", type=int, default=1)
    p_video.add_argument("--sk-start", type=int, default=1)
    p_video.add_argument("--sk-end", type=int, default=20)
    p_video.add_argument("--fps", type=float, default=4)
    add_mode_arg(p_video)
    p_video.add_argument("--mask", action="store_true", help="zero out background using a per-frame ch1 threshold mask")
    p_video.add_argument("--threshold", type=int, default=None, help="0-255 fixed threshold for --mask; default: Otsu (auto)")
    p_video.add_argument("--min-size", type=int, default=15, help="drop mask components smaller than this many pixels")
    p_video.add_argument("--open-radius", type=int, default=1, help="mask morphological opening kernel radius")
    p_video.add_argument("--gif", action="store_true", default=True)
    p_video.add_argument("--no-gif", dest="gif", action="store_false")
    p_video.add_argument("--mp4", action="store_true")
    p_video.add_argument("--out", type=Path)
    p_video.set_defaults(func=cmd_video)

    p_plate = sub.add_parser("plate", help="one composite thumbnail per well (plate overview)")
    p_plate.add_argument("--field", type=int, default=1)
    p_plate.add_argument("--sk", type=int, default=1)
    add_mode_arg(p_plate)
    p_plate.add_argument("--out", type=Path)
    p_plate.set_defaults(func=cmd_plate)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
