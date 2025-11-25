import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

gene = 'YNL118C'
color = 'black'
DB_PATH = f"db/{gene}_{color}.db"   # adjust if needed
CLIP_LEN = 193
N_CLIPS  = 10

# Stage boundaries (in frame index) and labels
STAGE_BOUNDARIES = [0, 64, 96, 128, 160, 192]
STAGE_LABELS     = [1, 2, 3, 4, 5, 6]

# --- helpers ---
def list_tables(conn):
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    ).fetchall()]

def table_columns(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info('{table}')").fetchall()]

def pick_area_column(conn):
    # Prefer this exact per-image mean column if available
    preferred = "AreaShape_Area"
    for t in list_tables(conn):
        cols = table_columns(conn, t)
        if preferred in cols:
            return t, preferred
    # Otherwise pick the first column containing 'AreaShape_Area'
    for t in list_tables(conn):
        cols = table_columns(conn, t)
        matches = [c for c in cols if "AreaShape_Area".lower() in c.lower()]
        if matches:
            return t, matches[0]
    raise RuntimeError("No column containing 'AreaShape_Area' found.")

def read_series(conn, table, col):
    return pd.read_sql_query(f"SELECT {col} FROM {table}", conn)[col].to_numpy()

# --- load data ---
with sqlite3.connect(DB_PATH) as conn:
    table, col = pick_area_column(conn)
    y = read_series(conn, table, col)

# --- plot 10 clips ---
fig, axes = plt.subplots(5, 2, figsize=(12, 12), sharey=True)
axes = axes.flatten()

total = len(y)
for i in range(N_CLIPS):
    start = i * CLIP_LEN
    stop  = min((i+1) * CLIP_LEN, total)
    ax = axes[i]
    if start >= total:
        ax.set_visible(False)
        continue

    # x indices within a clip
    x = np.arange(CLIP_LEN)
    ax.plot(x[:stop-start], y[start:stop])
    ax.set_xlim(0, CLIP_LEN - 1)

    # --- vertical dashed lines for stages ---
    for xb in STAGE_BOUNDARIES:
        ax.axvline(x=xb, linestyle='--', linewidth=0.8, color='gray')

    # --- red bold stage labels along the top (1..6) ---
    for j, label in enumerate(STAGE_LABELS):
        x_left = STAGE_BOUNDARIES[j]
        # right boundary for this stage (last stage just uses the end)
        if j + 1 < len(STAGE_BOUNDARIES):
            x_right = STAGE_BOUNDARIES[j + 1]
        else:
            x_right = CLIP_LEN - 1
        x_mid = 0.5 * (x_left + x_right)

    ax.set_title(f"Clip {i+1}")
    if i % 2 == 0:
        ax.set_ylabel("Area")  # or col if you prefer
    ax.set_xlabel("Frame")

fig.suptitle(f"{col} over 10 clips", y=0.995)
fig.tight_layout()
plt.savefig(f'area_plot_{gene}_{color}.png')
