import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from PIL import Image, ImageSequence
import io


def _find_red_center(red_img, prev_center=None):
    """Find the intersection point of x and y marginal peaks in the red channel.

    If multiple intersections exist, pick the one closest to prev_center.
    Returns (center_x, center_y).
    """
    x_marginal = red_img.mean(axis=0)
    y_marginal = red_img.mean(axis=1)

    x_peaks, _ = find_peaks(x_marginal)
    y_peaks, _ = find_peaks(y_marginal)
    if len(x_peaks) == 0:
        x_peaks = np.array([int(np.argmax(x_marginal))])
    if len(y_peaks) == 0:
        y_peaks = np.array([int(np.argmax(y_marginal))])

    # All intersection points
    intersections = [(xp, yp) for xp in x_peaks for yp in y_peaks]

    if prev_center is None or len(intersections) == 1:
        # Pick the intersection with the highest combined marginal intensity
        best = max(intersections, key=lambda p: x_marginal[p[0]] + y_marginal[p[1]])
    else:
        # Pick the intersection closest to previous frame's center
        best = min(intersections,
                   key=lambda p: (p[0] - prev_center[0])**2 + (p[1] - prev_center[1])**2)
    return best


def _find_cell_y_center(green_img, red_img):
    """Find the y-center of the whole cell from its y-boundary.

    Combines green and red channels, thresholds to find cell pixels,
    and returns the midpoint of (min_y, max_y).
    """
    combined = np.maximum(green_img, red_img).astype(float)
    # Project onto y-axis (sum across x)
    y_profile = combined.mean(axis=1)
    threshold = y_profile.max() * 0.1
    cell_rows = np.where(y_profile > threshold)[0]
    if len(cell_rows) == 0:
        return green_img.shape[0] // 2
    min_y = cell_rows[0]
    max_y = cell_rows[-1]
    return (min_y + max_y) // 2


def align_frames(green_frames, red_frames):
    """Align all frames: x uses red (nucleus) center, y uses whole-cell center.

    Returns aligned copies of green_frames and red_frames.
    """
    h, w = red_frames[0].shape
    cx, cy = w // 2, h // 2  # image center

    # First pass: find red centers (for x) and cell y-centers (for y)
    centers = []
    prev_center = None
    for i, red_img in enumerate(red_frames):
        center = _find_red_center(red_img, prev_center)
        cell_y = _find_cell_y_center(green_frames[i], red_img)
        centers.append((center[0], cell_y))
        prev_center = center

    # Second pass: shift both channels to align to image center
    aligned_green = []
    aligned_red = []
    for i, (center_x, center_y) in enumerate(centers):
        shift_x = cx - center_x
        shift_y = cy - center_y
        aligned_green.append(_shift_image(green_frames[i], shift_x, shift_y))
        aligned_red.append(_shift_image(red_frames[i], shift_x, shift_y))

    return aligned_green, aligned_red, centers


def _shift_image(img, shift_x, shift_y):
    """Shift a 2D image by (shift_x, shift_y) pixels, filling with zeros."""
    shifted = np.zeros_like(img)
    h, w = img.shape

    # Source and destination slicing
    src_y_start = max(0, -shift_y)
    src_y_end = min(h, h - shift_y)
    src_x_start = max(0, -shift_x)
    src_x_end = min(w, w - shift_x)

    dst_y_start = max(0, shift_y)
    dst_y_end = min(h, h + shift_y)
    dst_x_start = max(0, shift_x)
    dst_x_end = min(w, w + shift_x)

    shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        img[src_y_start:src_y_end, src_x_start:src_x_end]
    return shifted


def _draw_channel_panel(fig, gs_parent, channel_img, channel_name, color_fill,
                        color_line, cmap, global_max, frame_idx):
    """Draw a single channel's marginal plot (top + left marginals + center image + peak crosshair)."""
    gs = gs_parent.subgridspec(2, 2, width_ratios=[1, 4], height_ratios=[1, 4],
                                wspace=0.05, hspace=0.05)
    h, w = channel_img.shape

    # Marginal distributions
    x_marginal = channel_img.mean(axis=0)
    y_marginal = channel_img.mean(axis=1)

    # Find all local maxima (derivative zero-crossings, positive-to-negative)
    x_peaks, _ = find_peaks(x_marginal)
    y_peaks, _ = find_peaks(y_marginal)
    # Ensure at least the global max is included if no local peaks found
    if len(x_peaks) == 0:
        x_peaks = np.array([int(np.argmax(x_marginal))])
    if len(y_peaks) == 0:
        y_peaks = np.array([int(np.argmax(y_marginal))])

    # Top: x-axis marginal
    ax_top = fig.add_subplot(gs[0, 1])
    ax_top.fill_between(np.arange(w), x_marginal, color=color_fill, alpha=0.5)
    ax_top.plot(np.arange(w), x_marginal, color=color_line, linewidth=0.8)
    for xp in x_peaks:
        ax_top.axvline(xp, color='blue', linestyle='--', linewidth=0.9, alpha=0.8)
    ax_top.plot(x_peaks, x_marginal[x_peaks], 'v', color='blue',
                markersize=6, markeredgecolor='black', markeredgewidth=0.5)
    ax_top.set_xlim(0, w)
    ax_top.set_ylim(0, global_max)
    ax_top.set_xticks([])
    ax_top.set_ylabel('Intensity', fontsize=7)
    ax_top.tick_params(labelsize=6)

    # Left: y-axis marginal (rotated)
    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.fill_betweenx(np.arange(h), y_marginal, color=color_fill, alpha=0.5)
    ax_left.plot(y_marginal, np.arange(h), color=color_line, linewidth=0.8)
    for yp in y_peaks:
        ax_left.axhline(yp, color='blue', linestyle='--', linewidth=0.9, alpha=0.8)
    ax_left.plot(y_marginal[y_peaks], y_peaks, '>', color='blue',
                 markersize=6, markeredgecolor='black', markeredgewidth=0.5)
    ax_left.set_ylim(h, 0)
    ax_left.set_xlim(0, global_max)
    ax_left.set_yticks([])
    ax_left.set_xlabel('Intensity', fontsize=7)
    ax_left.tick_params(labelsize=6)

    # Center: channel image with all peak intersection crosshairs
    ax_img = fig.add_subplot(gs[1, 1])
    ax_img.imshow(channel_img, cmap=cmap, vmin=0, vmax=255)
    for xp in x_peaks:
        ax_img.axvline(xp, color='blue', linestyle='--', linewidth=0.8, alpha=0.7)
    for yp in y_peaks:
        ax_img.axhline(yp, color='blue', linestyle='--', linewidth=0.8, alpha=0.7)
    # Mark every intersection of x-peaks and y-peaks
    for xp in x_peaks:
        for yp in y_peaks:
            ax_img.plot(xp, yp, '+', color='blue', markersize=10, markeredgewidth=1.5)
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    # Corner label
    ax_corner = fig.add_subplot(gs[0, 0])
    ax_corner.text(0.5, 0.5, f'{channel_name}\nF{frame_idx}',
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   color=color_line)
    ax_corner.axis('off')


def make_marginal_gif(gene_name, method, base_dir="/datasets/yeast-imgs/generated_video"):
    gif_path = os.path.join(base_dir, gene_name, method, "0", "0_video.gif")

    if not os.path.exists(gif_path):
        print(f"File not found: {gif_path}")
        return

    img = Image.open(gif_path)
    rgb_frames = []
    green_frames = []
    red_frames = []
    for frame in ImageSequence.Iterator(img):
        frame_np = np.array(frame.convert('RGB'))
        rgb_frames.append(frame_np)
        green_frames.append(frame_np[:, :, 1])  # green channel
        red_frames.append(frame_np[:, :, 0])     # red channel

    print(f"Processing {gene_name} ({method}): {len(green_frames)} frames")

    # Align both channels based on the red channel intersection point
    green_frames, red_frames, centers = align_frames(green_frames, red_frames)
    print(f"Aligned frames to red channel centers: {centers}")

    # Build aligned RGB frames using the same shifts
    h, w = red_frames[0].shape
    cx, cy = w // 2, h // 2
    aligned_rgb_frames = []
    for i, (center_x, center_y) in enumerate(centers):
        shift_x = cx - center_x
        shift_y = cy - center_y
        aligned_rgb = np.stack([
            _shift_image(rgb_frames[i][:, :, c], shift_x, shift_y)
            for c in range(3)
        ], axis=-1)
        aligned_rgb_frames.append(Image.fromarray(aligned_rgb.astype(np.uint8)))

    # Global intensity range per channel for consistent axes
    green_max = max(f.mean(axis=0).max() for f in green_frames)
    green_max = max(green_max, max(f.mean(axis=1).max() for f in green_frames))
    red_max = max(f.mean(axis=0).max() for f in red_frames)
    red_max = max(red_max, max(f.mean(axis=1).max() for f in red_frames))

    output_frames = []

    for i in range(len(green_frames)):
        fig = plt.figure(figsize=(16, 8))
        gs_outer = gridspec.GridSpec(1, 2, wspace=0.15)

        _draw_channel_panel(fig, gs_outer[0], green_frames[i], 'Green',
                            color_fill='limegreen', color_line='darkgreen',
                            cmap='Greens', global_max=green_max, frame_idx=i)

        _draw_channel_panel(fig, gs_outer[1], red_frames[i], 'Red',
                            color_fill='salmon', color_line='darkred',
                            cmap='Reds', global_max=red_max, frame_idx=i)

        fig.suptitle(f"{gene_name} ({method})", fontsize=14, y=0.99)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        pil_frame = Image.open(buf).convert('RGB')
        output_frames.append(pil_frame)
        plt.close(fig)

    # Save marginal GIF
    out_path = f'./marginal_{gene_name}_{method}.gif'
    output_frames[0].save(
        out_path,
        save_all=True,
        append_images=output_frames[1:],
        duration=20,
        loop=0
    )
    print(f"Saved to {out_path}")

    # Save aligned RGB GIF
    aligned_out_path = f'./aligned_{gene_name}_{method}.gif'
    aligned_rgb_frames[0].save(
        aligned_out_path,
        save_all=True,
        append_images=aligned_rgb_frames[1:],
        duration=20,
        loop=0
    )
    print(f"Saved aligned RGB to {aligned_out_path}")


# Execution
target_gene = input("Enter Gene Name: ") or "YAL001C"
target_method = input("Enter Method: ") or "nucleus"
make_marginal_gif(target_gene, target_method)
