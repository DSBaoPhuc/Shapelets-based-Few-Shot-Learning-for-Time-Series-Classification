import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from numba import njit, prange
from scipy.stats import entropy
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from itertools import product
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
    # file_path = "../../data/BasicMotions/BasicMotions_TEST.ts"
    # file_path = "../../data/StandWalkJump/StandWalkJump_TRAIN.ts" # StandWalkJump
    # file_path = "../../data/Libras/Libras_TRAIN.ts" # Libras
    # file_path = "../../data/RacketSports/RacketSports_TRAIN.ts" # RacketSports
    # file_path = "../../data/Cricket/Cricket_TRAIN.ts" # Cricket
    # file_path = "../../data/Epilepsy/Epilepsy_TRAIN.ts" # Epilepsy
    # file_path = "../../data/ArticularyWordRecognition/ArticularyWordRecognition_TRAIN.ts" # ArticularyWordRecognition
    # file_path = "../../data/AtrialFibrillation/AtrialFibrillation_TRAIN.ts" # AtrialFibrillation
    # file_path = "../../data/AtrialFibrillation/AtrialFibrillation_TEST.ts" # AtrialFibrillation
    # file_path = "../../data/FingerMovements/FingerMovements_TRAIN.ts" # FingerMovements
    # file_path = "../../data/Heartbeat/Heartbeat_TRAIN.ts" # Heartbeat
    # file_path = "../../data/NATOPS/NATOPS_TRAIN.ts" # NATOPS
    # file_path = "../../data/SelfRegulationSCP1/SelfRegulationSCP1_TRAIN.ts" # SCP1
        
    
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


# @njit(fastmath=True)
# def subdist_fast(x, y):
#     s = 0.0
#     for i in range(x.shape[0]):
#         diff = x[i] - y[i]
#         s += diff * diff
#     return np.sqrt(s)

@njit(fastmath=True)
def subdist_fast(x, y):
    s = 0.0
    m = x.shape[0]
    for i in range(m):
        diff = x[i] - y[i]
        s += diff * diff
    # return np.sqrt(s)
    return np.sqrt(s / m)

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
        comp = 0.2*f_stat + 0.3*sep + 50 * ig 

        results.append((idx, series_id, start_pos, L,
                        true_class, predicted_class,
                        confidence, f_stat, sep, ig, comp))

    df = pd.DataFrame(results, columns=[
        'id', 'series_id', 'start', 'length',
        'true_class', 'pred_class', 'confidence',
        'F_stat', 'Separability', 'IG', 'Composite_Score'
    ])

    return df

# =========================
# CHUẨN HÓA METRICS
# =========================
def zscore_series(a):
    mu = a.mean()
    sigma = a.std()
    if sigma < 1e-12:
        return np.zeros_like(a)
    return (a - mu) / sigma


def normalize_metrics(df):
    df = df.copy()
    df["F_norm"] = zscore_series(df["F_stat"])
    df["Separability_norm"] = zscore_series(df["Separability"])
    df["IG_norm"] = zscore_series(df["IG"])
    return df


# =========================
# COMPOSITE SCORE w-F, w-sep, w-IG
# =========================
def compute_composite(df, w_F, w_sep, w_IG):
    return w_F * df["F_norm"] + w_sep * df["Separability_norm"] + w_IG * df["IG_norm"]


# =========================
# Shapelet Transform + SVM
# =========================
def shapelet_transform(shapelets, df_top, X):
    transform = np.zeros((X.shape[0], len(df_top)), dtype=np.float32)

    for i, sid in enumerate(df_top["id"].values):
        shapelet, _, _, _ = shapelets[sid]
        distances = compute_multivariate_distances(shapelet, X)
        transform[:, i] = distances

    return transform


# =========================
# GRID-SEARCH tối ưu trọng số
# =========================
def grid_search_weights(shapelets, df_norm, X, y, top_k=20,
                        weight_values=[0.1, 0.3, 0.5, 0.7, 1.0]):
    best_acc = -1
    best_weights = None
    best_df_top = None

    for w_F, w_sep, w_IG in product(weight_values, repeat=3):
        df = df_norm.copy()
        df["Composite_Score"] = compute_composite(df, w_F, w_sep, w_IG)
        df = df.sort_values(by="Composite_Score", ascending=False)

        df_top = df.head(top_k)
        X_trans = shapelet_transform(shapelets, df_top, X)

        clf = SVC(kernel="rbf")
        clf.fit(X_trans, y)
        y_pred = clf.predict(X_trans)
        acc = accuracy_score(y, y_pred)

        if acc > best_acc:
            best_acc = acc
            best_weights = (w_F, w_sep, w_IG)
            best_df_top = df_top

        print(f"Try w=({w_F},{w_sep},{w_IG}) → acc={acc:.4f}")

    print("\nBest weights:", best_weights, "→ accuracy:", best_acc)
    return best_df_top, best_weights, best_acc

def select_balanced_shapelets(df, y, top_k=20):
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)

    # Mỗi class lấy số shapelet gần bằng nhau
    k_per_class = top_k // num_classes
    remainder = top_k % num_classes

    selected_rows = []

    # Lấy balance theo true_class
    for c in unique_classes:
        df_c = df[df["true_class"] == c].sort_values("Composite_Score", ascending=False)

        k = k_per_class + (1 if remainder > 0 else 0)
        remainder -= 1

        selected_rows.append(df_c.head(k))

    # Ghép lại và giới hạn đúng top_k shapelets
    df_balanced = pd.concat(selected_rows).sort_values("Composite_Score", ascending=False).head(top_k)
    df_balanced.reset_index(drop=True, inplace=True)
    return df_balanced

def save_topk_balanced(shapelets, df, csv_path, num_classes=5, per_class=5):
    dfs = []

    for c in range(num_classes):
        df_c = df[df['true_class'] == c]\
                  .sort_values(by="Composite_Score", ascending=False)\
                  .head(per_class)
        dfs.append(df_c)

    df_balanced = pd.concat(dfs).sort_values(by="Composite_Score", ascending=False)

    shapelet_values = []
    for sid in df_balanced['id']:
        sl, _, _, _ = shapelets[sid]
        shapelet_values.append(sl.flatten().tolist())

    df_balanced["Shapelet_Values"] = shapelet_values
    df_balanced.to_csv(csv_path, index=False)

    print(f"Saved {per_class} shapelets per class → total {len(df_balanced)} → {csv_path}")
    return df_balanced

# ======================================
# MAIN PIPELINE
# ======================================
if __name__ == "__main__":

    X, y = load_data_multi()
    _, _, T = X.shape  # sequence length

    # Auto shapelet lengths
    L_min = int(0.1 * T)
    L_max = int(0.5 * T)
    L_step = max(5, int(0.05 * T))
    L_list = list(range(L_min, L_max + 1, L_step))

    num_per_length = 10

    shapelets = generate_multivariate_shapelets(X, y, L_list, num_per_length)
    df_scores = evaluate_multivariate_shapelets(shapelets, X, y)

    df_norm = normalize_metrics(df_scores)

    # Grid-search tối ưu trọng số
    df_best, best_weights, best_acc = grid_search_weights(
        shapelets, df_norm, X, y,
        top_k=20,
        weight_values=[0.1, 0.3, 0.5, 0.7, 1.0]
    )

    df_best["Shapelet_Values"] = [
        shapelets[sid][0].flatten().tolist()
        for sid in df_best["id"]
    ]

    best_shapelet = save_topk_balanced(shapelets, df_best, "top_shapelets_BM_opt.csv", num_classes=5, per_class=5)
    best_shapelet.to_csv("top_shapelets_BM_opt.csv", index=False)

    print("\nSaved optimized top-k shapelets →", "top_shapelets_BM_opt.csv")
    print("Best Weights:", best_weights, "Accuracy:", best_acc)