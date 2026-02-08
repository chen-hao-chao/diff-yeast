import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageSequence

def get_average_dynamics(gene_name, method, base_dir="/home/chchao0/projects/aip-rahulgk/chchao0/generated_video"):
    gif_paths = [
        os.path.join(base_dir, gene_name, method, str(i), f"{i}_video.gif") 
        for i in range(10)
    ]

    all_trajectories = []
    all_frames = [] 
    
    print(f"Processing {gene_name}...")

    for path in gif_paths:
        if not os.path.exists(path):
            continue
            
        img = Image.open(path)
        coords = []
        
        for frame in ImageSequence.Iterator(img):
            frame_np = np.array(frame.convert('RGB'))
            gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
            all_frames.append(gray)
            
            # Intensity-Weighted Centroid
            M = cv2.moments(gray)
            if M["m00"] != 0:
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]
                coords.append((cX, cY))
            else:
                coords.append(coords[-1] if coords else (0,0))
        all_trajectories.append(coords)

    if not all_trajectories:
        return

    # 1. Background Overlay
    background = np.max(np.stack(all_frames), axis=0)

    # 2. Average Trajectory
    max_len = max(len(t) for t in all_trajectories)
    processed_trajs = []
    for t in all_trajectories:
        xp = np.linspace(0, 1, len(t))
        xnew = np.linspace(0, 1, max_len)
        tx = np.interp(xnew, xp, [p[0] for p in t])
        ty = np.interp(xnew, xp, [p[1] for p in t])
        processed_trajs.append(np.stack([tx, ty], axis=1))

    mean_traj = np.mean(processed_trajs, axis=0)
    
    # 3. Plotting
    plt.figure(figsize=(12, 12))
    plt.imshow(background, cmap='gray', alpha=1.0)
    
    # Quiver Logic
    step = 3 
    X = mean_traj[:-step:step, 0]
    Y = mean_traj[:-step:step, 1]
    U = np.diff(mean_traj[::step, 0])
    V = np.diff(mean_traj[::step, 1])
    
    # Time mapping for colors (0 to 1)
    time_colors = np.linspace(0, 1, len(X))
    
    # 'plasma' is the Purple -> Pink -> Yellow map
    q = plt.quiver(X, Y, U, V, time_colors, 
                   angles='xy', scale_units='xy', scale=1, 
                   cmap='plasma', 
                   width=0.006, 
                   headwidth=4, 
                   headlength=5)
    
    # Optional: Mark exact Start and End
    plt.scatter(mean_traj[0,0], mean_traj[0,1], color='magenta', s=30, label='START', edgecolors='black')
    plt.scatter(mean_traj[-1,0], mean_traj[-1,1], color='yellow', s=30, label='END', edgecolors='black')

    cbar = plt.colorbar(q, shrink=0.5)
    cbar.set_label('Time Progression (Purple -> Yellow)')

    plt.title(f"Dynamic Trajectory: {gene_name} ({method})")
    plt.legend(loc='upper right')
    plt.savefig(f'./traj_{gene_name}_{method}_plasma.png', bbox_inches='tight', dpi=300)
    print(f"Saved to ./traj_{gene_name}_{method}_plasma.png")

# Execution
target_gene = input("Enter Gene Name: ") or "YAL001C"
target_method = input("Enter Method: ") or "nucleus"
get_average_dynamics(target_gene, target_method)