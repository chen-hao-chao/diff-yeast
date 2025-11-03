
# tsne_columns_as_samples.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pdb
# --- Load your CSV (each column is a feature vector) ---
option = 'green'
csv_path = f"results_IdentifyPrimaryGFP_{option}.csv"  # adjust path if needed
df = pd.read_csv(csv_path)

# Keep only numeric data (drop any non-numeric columns/rows just in case)
df_num = df.select_dtypes(include=[np.number])
if df_num.shape[1] == 0:
    # If headers made pandas treat the first row as header, try no-header read
    df = pd.read_csv(csv_path, header=None)
    df_num = df.apply(pd.to_numeric, errors="coerce")

# Clean NaNs (t-SNE can’t handle NaNs)
df_num = df_num.dropna(axis=0, how="all").dropna(axis=1, how="all")
if df_num.isna().any().any():
    df_num = df_num.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

# Each column = one sample; transpose for sklearn shape (n_samples, n_features)
X = df_num.to_numpy().T
n_samples = X.shape[0]

# --- Define your two sets (1-based indices as requested) ---
special_1b = np.array([1, 65, 97, 129, 161, 193], dtype=int)
special_0b = special_1b - 1
special_0b = special_0b[(special_0b >= 0) & (special_0b < n_samples)]

mask_special = np.zeros(n_samples, dtype=bool)
mask_special[special_0b] = True
mask_other = ~mask_special

# --- Run t-SNE ---
# Choose a safe perplexity (< n_samples), roughly ~min(30, (n-1)//3), but at least 5.
perplexity = max(5, min(30, max(2, (n_samples - 1) // 3)))
if perplexity >= n_samples:
    perplexity = max(2, n_samples // 3)

tsne = TSNE(
    n_components=2,
    init="pca",
    learning_rate="auto",
    random_state=42,
    perplexity=perplexity,
)
Y = tsne.fit_transform(X)

# --- Plot (let matplotlib choose different colors) ---
plt.figure(figsize=(7, 6))
plt.scatter(Y[mask_other, 0], Y[mask_other, 1], s=20, alpha=0.85, label="Other frames")
plt.scatter(Y[mask_special, 0], Y[mask_special, 1], s=40, alpha=0.95,
            label="Frames [1, 65, 97, 129, 161, 193]")

plt.title("t-SNE of Feature Vectors (columns as samples)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.tight_layout()
plt.savefig(f"tsne_results_columns_as_samples_{option}.png", dpi=150, bbox_inches="tight")
