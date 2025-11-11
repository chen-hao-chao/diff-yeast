import pandas as pd
import matplotlib.pyplot as plt

gene = 'YKR083C'
color = 'green'

# Load data
df = pd.read_csv(f"db/{gene}_{color}.db_frames_groups_coords.csv")

# Ensure ordered by time and keep first 193 "files"/frames
df = df.sort_values("row_index_1b")
traj = df[df["row_index_1b"] <= 193].copy()

# Split by label
real = traj[traj["group"] == "real"]
fake = traj[traj["group"] == "fake"]

# Extract coordinates for trajectory (in time order)
x = traj["PC1"].values
y = traj["PC2"].values
dx = x[1:] - x[:-1]
dy = y[1:] - y[:-1]

fig, ax = plt.subplots(figsize=(6, 6))

# 1) Scatter all points, colored by label
ax.scatter(real["PC1"], real["PC2"], s=25, alpha=0.7, label="real")
ax.scatter(fake["PC1"], fake["PC2"], s=25, alpha=0.7, label="fake", marker='x')

# 2) Draw arrows t -> t+1 along the ordered trajectory
ax.quiver(
    x[:-1], y[:-1],   # starting points
    dx, dy,           # directions
    angles='xy',
    scale_units='xy',
    scale=1,
    alpha=0.5,
)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(f"{gene} (first 193 frames, real vs fake)")
ax.set_aspect('equal', 'box')
ax.legend()
plt.tight_layout()
plt.savefig(f'seq_plot_{gene}_{color}.png', dpi=300)
