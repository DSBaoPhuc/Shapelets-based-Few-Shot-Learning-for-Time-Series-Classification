import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from numba import njit, prange
from scipy.stats import entropy

import os


# -----------------------------
# Load dataset from .ts file
# -----------------------------
def load_ts_file(file_path):
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    data_start = lines.index("@data") + 1
    X, y = [], []
    for line in lines[data_start:]:
        parts = line.split(":")
        dims = [np.array(list(map(float, dim.split(","))), dtype=np.float32)
                for dim in parts[:-1]]
        label = parts[-1].strip()
        X.append(dims)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = LabelEncoder().fit_transform(y)
    return X, y


def load_data_multi():
    file_path = "../../data/BasicMotions/BasicMotions_TRAIN.ts" # BasicMotions
    file_path = "../../data/StandWalkJump/StandWalkJump_TRAIN.ts" # StandWalkJump
    X, y = load_ts_file(file_path)

    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} dims, {X.shape[2]} steps.")
    print(f"Unique labels: {np.unique(y)}")

    # Normalize each dimension independently
    n_samples, n_dims, _ = X.shape
    for d in range(n_dims):
        scaler = StandardScaler()
        X[:, d, :] = scaler.fit_transform(X[:, d, :])

    return X, y


# -----------------------------
# Generate Shapelets
# -----------------------------
def generate_multivariate_shapelets(X, y, L_list, num_per_length):
    shapelets = []
    n_samples, n_dims, series_len = X.shape

    for i in range(n_samples):
        class_label = y[i]
        for L in L_list:
            max_start = series_len - L
            starts = np.random.choice(max_start, size=min(num_per_length, max_start), replace=False)
            for start in starts:
                sl = X[i, :, start:start + L]
                shapelets.append((sl, i, start, class_label))
    return shapelets


def euclidean_dist(S, T):
    return np.linalg.norm(S - T)


@njit(fastmath=True)
def z_norm_fast(ts):
    ts32 = ts.astype(np.float32)
    mean = np.mean(ts32)
    std = np.std(ts32)
    if std < 1e-8:
        return np.zeros(ts32.shape, dtype=np.float32)
    return ((ts32 - mean) / std).astype(np.float32)


@njit(fastmath=True)
def subdist_fast(x, y):
    s = 0.0
    for i in range(x.shape[0]):
        diff = x[i] - y[i]
        s += diff * diff
    return np.sqrt(s)


@njit(parallel=True, fastmath=True)
def compute_multivariate_distances(shapelet, X):
    n_samples, n_dims, series_len = X.shape
    m = shapelet.shape[1]
    distances = np.empty(n_samples, dtype=np.float32)

    for i in prange(n_samples):
        best = np.inf
        for start in range(series_len - m + 1):
            d_sum = 0.0
            for d in range(n_dims):
                sub = X[i, d, start:start + m]
                s_d = z_norm_fast(shapelet[d])
                sub_d = z_norm_fast(sub)
                diff = s_d - sub_d
                d_sum += np.sqrt(np.sum(diff * diff))
            dist = d_sum / n_dims
            if dist < best:
                best = dist
        distances[i] = best
    return distances


def information_gain(distances, labels):
    idx = np.argsort(distances)
    sorted_d = distances[idx]
    sorted_y = labels[idx]
    n = len(distances)

    def entropy_from_counts(counts):
        probs = counts / np.sum(counts)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    _, counts_total = np.unique(labels, return_counts=True)
    H_total = entropy_from_counts(counts_total)

    best_ig = 0.0
    for s in range(1, n):
        left_y = sorted_y[:s]
        right_y = sorted_y[s:]

        _, left_counts = np.unique(left_y, return_counts=True)
        _, right_counts = np.unique(right_y, return_counts=True)

        H_left = entropy_from_counts(left_counts) if left_y.size > 0 else 0.0
        H_right = entropy_from_counts(right_counts) if right_y.size > 0 else 0.0

        ig = H_total - (s / n) * H_left - ((n - s) / n) * H_right
        if ig > best_ig:
            best_ig = ig

    return float(best_ig)

def evaluate_multivariate_shapelets(shapelets, X, y):
    results = []
    unique_classes = np.unique(y)

    for idx, (shapelet, series_id, start_pos, true_class) in enumerate(tqdm(shapelets, desc="Evaluating")):
        L = shapelet.shape[1]
        distances = compute_multivariate_distances(shapelet, X)

        # F-stat
        dist_class = [distances[y == c] for c in unique_classes]
        f_stat, _ = f_oneway(*dist_class)

        # Separability
        class_mean = np.array([dc.mean() for dc in dist_class])
        sep = class_mean.max() - class_mean.min()

        # IG based on orderline split
        ig = information_gain(distances, y)

        # Predicted class
        predicted_class = unique_classes[np.argmin(class_mean)]
        sorted_means = np.sort(class_mean)
        confidence = sorted_means[1] - sorted_means[0]

        # Composite Score upgraded with IG
        comp = f_stat + sep + 50 * ig  # alpha = 50 (tunable weight)

        results.append((idx, series_id, start_pos, L,
                        true_class, predicted_class,
                        confidence, f_stat, sep, ig, comp))

    df = pd.DataFrame(results, columns=[
        'id', 'series_id', 'start', 'length',
        'true_class', 'pred_class', 'confidence',
        'F_stat', 'Separability', 'IG', 'Composite_Score'
    ])

    return df


# -----------------------------
# Save Top-K Balanced
# -----------------------------
def save_topk_balanced(shapelets, df, top_k, csv_path, num_classes=4):

    per_class = top_k // num_classes
    dfs = []

    for c in range(num_classes):
        df_c = df[df['true_class'] == c].sort_values(by="Composite_Score", ascending=False).head(per_class)
        dfs.append(df_c)

    df_balanced = pd.concat(dfs).sort_values(by="Composite_Score", ascending=False)

    shapelet_values = []
    for sid in df_balanced['id']:
        sl, _, _, _ = shapelets[sid]
        shapelet_values.append(sl.flatten().tolist())

    df_balanced["Shapelet_Values"] = shapelet_values
    df_balanced.to_csv(csv_path, index=False)

    print(f"Balanced Top-{top_k} saved → {csv_path}")
    return df_balanced


# -----------------------------
# Visualization
# -----------------------------
def plot_shapelet_on_series(shapelet, X, series_id, start_pos, label, idx):
    dims, L = shapelet.shape
    series = X[series_id]

    plt.figure(figsize=(12, 5))

    for d in range(dims):
        plt.plot(series[d], alpha=0.25)

    for d in range(dims):
        plt.plot(range(start_pos, start_pos + L), shapelet[d], linewidth=2)

    plt.title(f"Shapelet #{idx} | class={label} | start={start_pos} | L={L}")
    plt.show()


# =======================================
# MAIN PIPELINE
# =======================================
if __name__ == "__main__":

    X, y = load_data_multi()

    L_list = [20, 30, 40]
    num_per_length = 20
    top_k = 12  # Must be divisible by number of classes (4)

    shapelets = generate_multivariate_shapelets(X, y, L_list, num_per_length)

    df_scores = evaluate_multivariate_shapelets(shapelets, X, y)

    df_topk = save_topk_balanced(shapelets, df_scores, top_k, "top_shapelets_balanced_SWJ.csv")

    # print("Visualizing top shapelets...")
    # for idx, row in df_topk.iterrows():
    #     sid = row["id"]
    #     shapelet, series_id, start, label = (shapelets[sid][0],
    #                                          shapelets[sid][1],
    #                                          shapelets[sid][2],
    #                                          shapelets[sid][3])

    #     plot_shapelet_on_series(shapelet, X, series_id, start, label, idx)
