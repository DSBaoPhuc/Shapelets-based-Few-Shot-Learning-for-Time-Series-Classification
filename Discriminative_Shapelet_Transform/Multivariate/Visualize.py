import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# --- Load function ---
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


# --- Subdist: khoảng cách giữa shapelet và đoạn con ---
def subdist(series, shapelet):
    n_dims, series_len = series.shape
    L = shapelet.shape[1]
    min_dist = np.inf
    best_start = 0
    for i in range(series_len - L + 1):
        window = series[:, i:i+L]
        dist = np.linalg.norm(window - shapelet)
        if dist < min_dist:
            min_dist = dist
            best_start = i
    return min_dist, best_start


# --- Parse shapelet values ---
def parse_shapelet_values(value_str):
    dims = value_str.strip('"').split(";")
    arr = [np.array(list(map(float, dim.split(","))), dtype=np.float32) for dim in dims]
    return np.array(arr)


file_path = "../../data/BasicMotions/BasicMotions_TRAIN.ts"
X, y = load_ts_file(file_path)

df = pd.read_csv("top_shapelets_multi.csv")

# --- Chọn sample đầu tiên ---
sample_idx = 0
series = X[sample_idx]  # (6, 100)

for idx, row in df.iterrows():
    shapelet = parse_shapelet_values(row["Shapelet_Values"])
    L = int(row["length"])
    dims = shapelet.shape[0]

    # Find the best matching position
    _, best_start = subdist(series, shapelet)

    plt.figure(figsize=(12,6))
    for i in range(dims):
        plt.plot(series[i], label=f"dim {i}", alpha=0.7)
        plt.plot(range(best_start, best_start+L), series[i, best_start:best_start+L], 
                 linewidth=3, label=f"Shapelet match (dim {i})")

    plt.title(f"Shapelet #{row['id']} matched at pos={best_start} | len={L}")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()
