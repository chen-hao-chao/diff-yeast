# frames_groups_pca.py
import sqlite3, pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

DB_PATH = "db/YNL118C_green.db"  # adjust path if needed
RANDOM_STATE = 42
BASE   = np.array([1, 65, 97, 129, 161, 193], dtype=int)
STRIDE = 193
GROUPS = list(range(1, 11))  # i = 1..10

def list_tables(conn):
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    ).fetchall()]

def load_table(conn, table, limit=None):
    q = f"SELECT * FROM {table}" + (f" LIMIT {int(limit)}" if limit else "")
    return pd.read_sql_query(q, conn)

def numeric_df(df):
    df_num = df.select_dtypes(include=[np.number])
    if df_num.shape[1] == 0:
        df_num = df.apply(pd.to_numeric, errors="coerce")
    df_num = df_num.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df_num.isna().any().any():
        df_num = df_num.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return df_num

def pick_numeric_table(conn):
    best, best_cols = None, -1
    for t in list_tables(conn):
        df = load_table(conn, t, limit=2000)
        df_num = numeric_df(df)
        if df_num.shape[1] > best_cols and df_num.shape[0] > 0 and df_num.shape[1] > 0:
            best, best_cols = t, df_num.shape[1]
    if best is None:
        raise RuntimeError("No table with numeric data found.")
    return best

def build_real_mask(n_rows):
    # 1-based indices for all selected sets
    idxs = []
    for i in GROUPS:
        idx_1b = BASE + (i - 1) * STRIDE
        idx_1b = idx_1b[(idx_1b >= 1) & (idx_1b <= n_rows)]
        idxs.extend(idx_1b.tolist())

    mask_real = np.zeros(n_rows, dtype=bool)
    if idxs:
        mask_real[(np.array(idxs) - 1).astype(int)] = True
    mask_fake = ~mask_real
    return mask_real, mask_fake


def plot_groups(X2, mask_real, mask_fake, title, out_path):
    plt.figure(figsize=(8, 7))
    # Fake first (blue)
    plt.scatter(X2[mask_fake, 0], X2[mask_fake, 1], s=12, alpha=0.7, label="fake")
    # Real second (orange)
    plt.scatter(X2[mask_real, 0], X2[mask_real, 1], s=24, alpha=0.95, label="real")
    plt.title(title)
    plt.xlabel("PC 1"); plt.ylabel("PC 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def main():
    conn = sqlite3.connect(DB_PATH)
    table = pick_numeric_table(conn)
    df = load_table(conn, table)
    M = numeric_df(df)          # rows × features
    X = M.to_numpy()
    n_rows = X.shape[0]

    # PCA → 2D
    X2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)

    # Build two-color masks
    mask_real, mask_fake = build_real_mask(n_rows)

    # Plot + save
    fig_path = Path(DB_PATH + "_frames_groups_pca.png")
    plot_groups(X2, mask_real, mask_fake, f"PCA — rows as samples — table: {table}", fig_path)

    # Save coordinates + binary group label
    labels = np.where(mask_real, "real", "fake")
    coords = pd.DataFrame({
        "row_index_1b": np.arange(1, n_rows+1, dtype=int),
        "PC1": X2[:,0],
        "PC2": X2[:,1],
        "group": labels,
    })
    coords.to_csv(DB_PATH + "_frames_groups_coords.csv", index=False)
    print("Saved:", fig_path, "and", DB_PATH + "_frames_groups_coords.csv")

if __name__ == "__main__":
    main()
