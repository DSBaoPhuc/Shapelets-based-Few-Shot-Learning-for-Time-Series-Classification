import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit, prange
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder
import time


# ===========================
# Load multivariate dataset (.ts)
# ===========================

def load_ts_file(file_path):
    """
    Đọc file .ts định dạng UEA/UCR multivariate.
    Trả về:
        X: numpy array (n_samples, n_dimensions, series_length)
        y: numpy array (n_samples,)
    """
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
    file_path = "../../data/BasicMotions/BasicMotions_TRAIN.ts"
    X, y = load_ts_file(file_path)
    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} dimensions, {X.shape[2]} timesteps.")
    print(f"Unique labels: {np.unique(y)}")
    return X, y


# ===========================
# Shapelet utilities
# ===========================

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
    """
    Tính khoảng cách giữa 1 shapelet đa chiều và tất cả các chuỗi X.
    shapelet: (n_dims, L)
    X: (n_samples, n_dims, series_len)
    """
    n_samples, n_dims, series_len = X.shape
    m = shapelet.shape[1]
    distances = np.empty(n_samples, dtype=np.float32)

    for i in prange(n_samples):
        best = np.inf
        for start in range(0, series_len - m + 1):
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
        e = 0.0
        for p in probs:
            if p > 0:
                e -= p * np.log2(p)
        return e

    unique, counts = np.unique(labels, return_counts=True)
    H_total = entropy_from_counts(counts)

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


def separability_score(distances, labels):
    unique = np.unique(labels)
    if unique.shape[0] < 2:
        return 0.0
    means = [np.mean(distances[labels == lbl]) for lbl in unique]
    mean_diff = np.max(means) - np.min(means)
    pooled_std = np.std(distances) + 1e-8
    return float(mean_diff / pooled_std)


def f_statistic_score(distances, labels):
    groups = [distances[labels == lbl] for lbl in np.unique(labels)]
    if len(groups) > 1:
        try:
            f_val, _ = f_oneway(*groups)
            return float(f_val)
        except Exception:
            return 0.0
    return 0.0


# ===========================
# Multivariate Shapelet Extraction
# ===========================

def generate_multivariate_shapelets(X, min_len=10, max_len=50, step=5, max_per_series=None):
    """
    Sinh shapelet đa chiều (mỗi shapelet có đủ n_dims).
    """
    candidates = []
    n_samples, n_dims, series_len = X.shape
    for i in range(n_samples):
        for L in range(min_len, min(max_len, series_len) + 1, step):
            stride = max(1, L // 2)
            for start in range(0, series_len - L + 1, stride):
                sl = X[i, :, start:start + L].astype(np.float32).copy()  # (n_dims, L)
                candidates.append(sl)
                if max_per_series is not None and len(candidates) >= max_per_series:
                    return candidates
    return candidates


def evaluate_shapelets_multi(shapelets, X, y, weights=(0.4, 0.3, 0.3)):
    w_ig, w_f, w_sep = weights
    results = []

    for i, shapelet in enumerate(tqdm(shapelets, desc="Evaluating multivariate shapelets", leave=False)):
        distances = compute_multivariate_distances(shapelet, X)
        ig = information_gain(distances, y)
        f_stat = f_statistic_score(distances, y)
        sep = separability_score(distances, y)
        comp = w_ig * ig + w_f * np.log1p(abs(f_stat)) + w_sep * sep
        results.append((i, shapelet.shape[1], ig, f_stat, sep, comp))

    df = pd.DataFrame(results, columns=['id', 'length', 'IG', 'F_stat', 'Separability', 'Composite_Score'])
    df.sort_values(by='Composite_Score', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def save_topk_shapelets(df, shapelets, k=10, out_file="top_shapelets_multi.csv"):
    top_df = df.head(k).copy()
    top_shapelets = [shapelets[idx] for idx in top_df['id']]
    top_df['Shapelet_Values'] = [
        ";".join([",".join(map(lambda x: f"{x:.4f}", s)) for s in shapelet]) for shapelet in top_shapelets
    ]
    top_df.to_csv(out_file, index=False)
    print(f"Saved top-{k} multivariate shapelets to: {out_file}")


# ===========================
# Main
# ===========================

if __name__ == "__main__":
    start_time = time.time()
    X, y = load_data_multi()

    print("\nGenerating multivariate shapelet candidates ...")
    candidates = generate_multivariate_shapelets(X, min_len=10, max_len=40, step=10, max_per_series=200)
    print(f"Total multivariate candidates: {len(candidates)}")

    print("\nEvaluating shapelets ...")
    df_results = evaluate_shapelets_multi(candidates, X, y)
    print(df_results.head(10))

    save_topk_shapelets(df_results, candidates, k=10, out_file="top_shapelets_multi.csv")
    print(f"\nDone in {time.time() - start_time:.2f} seconds.")
