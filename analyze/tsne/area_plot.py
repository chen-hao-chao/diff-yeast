import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DB_PATH = "YDR458C_green.db"   # adjust if needed
CLIP_LEN = 193
N_CLIPS  = 10

# --- helpers ---
def list_tables(conn):
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    ).fetchall()]

def table_columns(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info('{table}')").fetchall()]

def pick_area_column(conn):
    # Prefer this exact per-image mean column if available
    preferred = "Mean_IdentifyPrimaryGFP_AreaShape_Area"
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
    x = np.arange(start + 1, stop + 1)  # 1-based frame numbering
    ax.plot(x, y[start:stop])
    ax.set_title(f"Clip {i+1}: frames {start+1}–{stop}")
    if i % 2 == 0:
        ax.set_ylabel(col)
    ax.set_xlabel("Frame")

fig.suptitle(f"{col} over 10 clips (length={CLIP_LEN}) — table: {table}", y=0.995)
fig.tight_layout()
plt.savefig('area_plot.png')
