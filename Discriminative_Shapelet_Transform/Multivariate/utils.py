import numpy as np
import pandas as pd
import json
import os
from numba import njit, prange

# -----------------------------
# Utils
# -----------------------------
@njit(fastmath=True)
def z_norm_window(x):
    mean = x.mean()
    std = x.std()
    if std < 1e-8:
        return np.zeros(x.shape, dtype=np.float32)
    return ((x - mean) / std).astype(np.float32)

@njit(fastmath=True)
def euclid_1d(a, b):
    s = 0.0
    for i in range(a.shape[0]):
        diff = a[i] - b[i]
        s += diff * diff
    return np.sqrt(s)

@njit(fastmath=True)
def window_distance(shapelet, subseq):
    n_dims = shapelet.shape[0]
    m = shapelet.shape[1]

    d_sum = 0.0
    for d in range(n_dims):
        s_d = z_norm_window(shapelet[d].astype(np.float32))
        sub_d = z_norm_window(subseq[d].astype(np.float32))
        d_sum += euclid_1d(s_d, sub_d)

    return d_sum / n_dims   # mean across dimensions

@njit(parallel=True, fastmath=True)
def compute_multivariate_distances(shapelet, X):
    n_samples, n_dims, series_len = X.shape
    m = shapelet.shape[1]
    distances = np.empty(n_samples, dtype=np.float32)

    for i in prange(n_samples):
        best = np.inf

        for start in range(series_len - m + 1):
            subseq = X[i, :, start:start + m]     # shape (n_dims, m)
            dist = window_distance(shapelet, subseq)

            if dist < best:
                best = dist
        
        distances[i] = best

    return distances


# -----------------------------
# Save Shapelets to file
# -----------------------------
def save_topk_balanced(shapelets, df, csv_path, num_classes=5, per_class=5):
    """
    Save top-k shapelets per class, each dimension saved in a separate column (dim_0, dim_1, ...).
    Each cell contains a JSON string of the list of values for that dimension.
    """
    dfs = []

    for c in range(num_classes):
        df_c = df[df['true_class'] == c]\
                  .sort_values(by="Composite_Score", ascending=False)\
                  .head(per_class)
        dfs.append(df_c)

    df_balanced = pd.concat(dfs).sort_values(by="Composite_Score", ascending=False).reset_index(drop=True)

    # Find the maximum number of dimensions among the selected shapelets (to create enough dim_i columns)
    selected_ids = df_balanced['id'].tolist()
    max_dims = 0
    for sid in selected_ids:
        sl, _, _, _ = shapelets[sid]
        if sl.ndim >= 1:
            max_dims = max(max_dims, sl.shape[0])

    dim_columns = [f"dim_{d}" for d in range(max_dims)]

    # Get per-dimension values for each shapelet
    rows_dim_values = {col: [] for col in dim_columns}
    shapelet_lengths = []  # length L of each shapelet
    for sid in selected_ids:
        sl, _, _, _ = shapelets[sid]  # sl shape: (n_dims, L)
        n_dims = sl.shape[0]
        L = sl.shape[1]
        shapelet_lengths.append(L)

        for d in range(max_dims):
            if d < n_dims:
                # save as JSON string to safely write to CSV
                rows_dim_values[f"dim_{d}"].append(json.dumps(sl[d].tolist()))
            else:
                # If shapelet does not have this dimension -> save empty list
                rows_dim_values[f"dim_{d}"].append(json.dumps([]))

    # add dimension columns to dataframe
    for col in dim_columns:
        df_balanced[col] = rows_dim_values[col]
    df_balanced['Shapelet_Length'] = shapelet_lengths

    # Save as CSV
    dir_path = os.path.dirname(csv_path)
    if dir_path not in ["", "."]:
        os.makedirs(dir_path, exist_ok=True)

    df_balanced.to_csv(csv_path, index=False)

    print(f"Saved {per_class} shapelets per class → total {len(df_balanced)} → {csv_path}")
    return df_balanced