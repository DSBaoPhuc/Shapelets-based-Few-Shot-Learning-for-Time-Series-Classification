import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from numba import njit, prange
from scipy.stats import entropy
import json
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
    # file_path = "../../data/StandWalkJump/StandWalkJump_TRAIN.ts" # StandWalkJump
    # file_path = "../../data/Libras/Libras_TRAIN.ts" # Libras
    # file_path = "../../data/RacketSports/RacketSports_TRAIN.ts" # RacketSports
    # file_path = "../../data/Cricket/Cricket_TRAIN.ts" # Cricket
    # file_path = "../../data/Epilepsy/Epilepsy_TRAIN.ts" # Epilepsy
    # file_path = "../../data/ArticularyWordRecognition/ArticularyWordRecognition_TRAIN.ts" # ArticularyWordRecognition
    # file_path = "../../data/AtrialFibrillation/AtrialFibrillation_TRAIN.ts" # AtrialFibrillation
    # file_path = "../../data/FingerMovements/FingerMovements_TRAIN.ts" # FingerMovements
    # file_path = "../../data/Heartbeat/Heartbeat_TRAIN.ts" # Heartbeat
    # file_path = "../../data/NATOPS/NATOPS_TRAIN.ts" # NATOPS
    # file_path = "../../data/SelfRegulationSCP1/SelfRegulationSCP1_TRAIN.ts" # SCP1
        
        
    #Test files
    # file_path = "../../data/BasicMotions/BasicMotions_TEST.ts"
    # file_path = "../../data/Cricket/Cricket_TEST.ts"
    # file_path = "../../data/Epilepsy/Epilepsy_TEST.ts"
    # file_path = "../../data/AtrialFibrillation/AtrialFibrillation_TEST.ts" # AtrialFibrillation
    
    
    X, y = load_ts_file(file_path)

    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} dims, length {X.shape[2]}.") 
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
                sl = X[i, :, start:start + L].astype(np.float32)
                shapelets.append((sl, i, start, class_label))
    return shapelets

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
        comp = 20*f_stat + 30*sep + 50*ig 

        results.append((idx, series_id, start_pos, L,
                        true_class, predicted_class,
                        confidence, f_stat, sep, ig, comp))

    df = pd.DataFrame(results, columns=[
        'id', 'series_id', 'start', 'length',
        'true_class', 'pred_class', 'confidence',
        'F_stat', 'Separability', 'IG', 'Composite_Score'
    ])

    return df


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
    _, _, T = X.shape  # sequence length

    L_min = int(0.1 * T)  # 10% length of T
    L_max = int(0.75 * T)  # 75% length of T
    L_step = max(5, int(0.05 * T))  # step size 5% of T or 5 if T is small

    L_list = list(range(L_min, L_max + 1, L_step))
    print(f"\n Auto Generated Shapelet Lengths: {L_list}\n")

    num_per_length = 10
    
    shapelets = generate_multivariate_shapelets(X, y, L_list, num_per_length)

    df_scores = evaluate_multivariate_shapelets(shapelets, X, y)

    df_topk = save_topk_balanced(
        shapelets, df_scores, "shapelets_BasicMotions.csv",
        num_classes=len(np.unique(y)), per_class=5
    )
