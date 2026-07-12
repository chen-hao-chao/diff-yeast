import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageSequence
from scipy.ndimage import gaussian_filter1d


# --- Tunable Parameters ---
SEARCH_RADIUS = 1.25       # max pixels to search from current position each step
MAX_ANGLE_DEG = 40      # max deviation from current heading per step
SMOOTH_SIGMA = 0.1        # Gaussian smoothing strength
QUIVER_STEP = 3         # arrow spacing (1 arrow per N trajectory points)
ARROW_WIDTH = 0.006     # quiver arrow thickness


def trace_directed_trajectory(frames_red, start_pos, initial_heading,
                              search_radius=SEARCH_RADIUS,
                              max_angle_deg=MAX_ANGLE_DEG):
    """Trace a trajectory by following the highest red intensity within
    a cone of heading ± max_angle_deg, up to search_radius pixels away.

    Args:
        frames_red: list of red channel arrays (float64)
        start_pos: (x, y) starting position (shared for both cells)
        initial_heading: angle in radians (0=right, pi=left)
        search_radius: how far to look each step
        max_angle_deg: allowed cone half-angle from current heading
    """
    max_angle = np.radians(max_angle_deg)
    coords = [start_pos]
    heading = initial_heading

    for frame_idx in range(1, len(frames_red)):
        red = frames_red[frame_idx]
        h, w = red.shape
        cx, cy = coords[-1]

        best_val = -1
        best_pos = (cx, cy)

        y_min = max(0, int(cy - search_radius))
        y_max = min(h, int(cy + search_radius + 1))
        x_min = max(0, int(cx - search_radius))
        x_max = min(w, int(cx + search_radius + 1))

        for py in range(y_min, y_max):
            for px in range(x_min, x_max):
                dx = px - cx
                dy = py - cy
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1.0 or dist > search_radius:
                    continue

                # Angle constraint: must be within cone
                pixel_angle = np.arctan2(dy, dx)
                angle_diff = np.arctan2(np.sin(pixel_angle - heading),
                                        np.cos(pixel_angle - heading))
                if abs(angle_diff) > max_angle:
                    continue

                val = red[py, px]
                if val > best_val:
                    best_val = val
                    best_pos = (float(px), float(py))

        # Update heading toward the chosen position
        dx = best_pos[0] - cx
        dy = best_pos[1] - cy
        if dx*dx + dy*dy > 0.5:
            heading = np.arctan2(dy, dx)

        coords.append(best_pos)

    return coords


def smooth_trajectory(traj, sigma=SMOOTH_SIGMA):
    """Gaussian-smooth a trajectory to remove jitter."""
    smoothed = np.copy(traj)
    smoothed[:, 0] = gaussian_filter1d(traj[:, 0], sigma=sigma)
    smoothed[:, 1] = gaussian_filter1d(traj[:, 1], sigma=sigma)
    return smoothed


def get_average_dynamics(gene_name, method, base_dir="/datasets/yeast-imgs/generated_video"):
    gif_paths = [
        os.path.join(base_dir, gene_name, method, str(i), f"{i}_video.gif")
        for i in range(10)
    ]

    all_mother_trajs = []
    all_daughter_trajs = []
    all_frames = []

    print(f"Processing {gene_name}...")

    for path in gif_paths:
        if not os.path.exists(path):
            continue

        img = Image.open(path)
        frames_red = []
        frames_gray = []

        for frame in ImageSequence.Iterator(img):
            frame_np = np.array(frame.convert('RGB'))
            gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
            red = frame_np[:, :, 0].astype(np.float64)
            frames_gray.append(gray)
            frames_red.append(red)

        all_frames.extend(frames_gray)

        # Shared start: intensity-weighted centroid of red channel in first frame
        M = cv2.moments(frames_red[0])
        if M["m00"] != 0:
            start_pos = (M["m10"] / M["m00"], M["m01"] / M["m00"])
        else:
            h, w = frames_red[0].shape
            start_pos = (w / 2.0, h / 2.0)

        # Mother goes RIGHT (heading=0), Daughter goes LEFT (heading=pi)
        mother_coords = trace_directed_trajectory(
            frames_red, start_pos, initial_heading=0)
        daughter_coords = trace_directed_trajectory(
            frames_red, start_pos, initial_heading=np.pi)

        all_mother_trajs.append(mother_coords)
        all_daughter_trajs.append(daughter_coords)

    if not all_mother_trajs:
        return

    # 1. Background Overlay
    background = np.max(np.stack(all_frames), axis=0)

    # 2. Interpolate & average Mother trajectories
    max_len = max(len(t) for t in all_mother_trajs)
    processed_mother = []
    for t in all_mother_trajs:
        xp = np.linspace(0, 1, len(t))
        xnew = np.linspace(0, 1, max_len)
        tx = np.interp(xnew, xp, [p[0] for p in t])
        ty = np.interp(xnew, xp, [p[1] for p in t])
        processed_mother.append(np.stack([tx, ty], axis=1))
    mean_mother = np.mean(processed_mother, axis=0)

    # 3. Interpolate & average Daughter trajectories
    processed_daughter = []
    for t in all_daughter_trajs:
        xp = np.linspace(0, 1, len(t))
        xnew = np.linspace(0, 1, max_len)
        tx = np.interp(xnew, xp, [p[0] for p in t])
        ty = np.interp(xnew, xp, [p[1] for p in t])
        processed_daughter.append(np.stack([tx, ty], axis=1))
    mean_daughter = np.mean(processed_daughter, axis=0)

    # 4. Smooth
    mean_mother = smooth_trajectory(mean_mother)
    mean_daughter = smooth_trajectory(mean_daughter)

    # 5. Plotting
    plt.figure(figsize=(12, 12))
    plt.imshow(background, cmap='gray', alpha=1.0)

    step = QUIVER_STEP

    # --- Mother Trajectory (plasma: purple -> yellow, goes RIGHT) ---
    X = mean_mother[:-step:step, 0]
    Y = mean_mother[:-step:step, 1]
    U = np.diff(mean_mother[::step, 0])
    V = np.diff(mean_mother[::step, 1])
    time_colors = np.linspace(0, 1, len(X))

    q_mother = plt.quiver(X, Y, U, V, time_colors,
                   angles='xy', scale_units='xy', scale=1,
                   cmap='plasma',
                   width=ARROW_WIDTH,
                   headwidth=4,
                   headlength=5)

    plt.scatter(mean_mother[-1, 0], mean_mother[-1, 1],
                color='yellow', s=30, label='Mother END', edgecolors='black')

    cbar_mother = plt.colorbar(q_mother, shrink=0.4, pad=0.02)
    cbar_mother.set_label('Mother (right): Time (Purple → Yellow)')

    # --- Daughter Trajectory (cool: cyan -> blue, goes LEFT) ---
    Xd = mean_daughter[:-step:step, 0]
    Yd = mean_daughter[:-step:step, 1]
    Ud = np.diff(mean_daughter[::step, 0])
    Vd = np.diff(mean_daughter[::step, 1])
    time_colors_d = np.linspace(0, 1, len(Xd))

    q_daughter = plt.quiver(Xd, Yd, Ud, Vd, time_colors_d,
                   angles='xy', scale_units='xy', scale=1,
                   cmap='cool',
                   width=ARROW_WIDTH,
                   headwidth=4,
                   headlength=5)

    plt.scatter(mean_daughter[-1, 0], mean_daughter[-1, 1],
                color='blue', s=30, label='Daughter END', edgecolors='black')

    cbar_daughter = plt.colorbar(q_daughter, shrink=0.4, pad=0.08)
    cbar_daughter.set_label('Daughter (left): Time (Cyan → Blue)')

    # Shared start marker
    plt.scatter(mean_mother[0, 0], mean_mother[0, 1],
                color='white', s=50, label='START (shared)', edgecolors='black', zorder=5)

    plt.title(f"Mother/Daughter Trajectory: {gene_name} ({method})")
    plt.legend(loc='upper right')
    plt.savefig(f'./traj_{gene_name}_{method}_dual.png', bbox_inches='tight', dpi=300)
    print(f"Saved to ./traj_{gene_name}_{method}_dual.png")

# Execution
target_gene = input("Enter Gene Name: ") or "YAL001C"
target_method = input("Enter Method: ") or "nucleus"
get_average_dynamics(target_gene, target_method)
