import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import f_oneway
from utils import compute_multivariate_distances

# -----------------------------
# Evaluate Shapelets
# -----------------------------
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
