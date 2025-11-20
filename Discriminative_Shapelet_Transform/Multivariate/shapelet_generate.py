import numpy as np

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