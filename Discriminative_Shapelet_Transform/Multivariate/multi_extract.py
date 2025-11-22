import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

from shapelet_generate import generate_multivariate_shapelets
from evaluate import evaluate_multivariate_shapelets
from utils import save_topk_balanced


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
    # file_path = "../../data/BasicMotions/BasicMotions_TRAIN.ts" # BasicMotions
    # file_path = "../../data/Handwriting/Handwriting_TRAIN.ts" # Handwriting
    # file_path = "../../data/JapaneseVowels/JapaneseVowels_eq_TRAIN.ts" # JapaneseVowels
    # file_path = "../../data/UWaveGestureLibrary/UWaveGestureLibrary_TRAIN.ts" # UWaveGestureLibrary   
    # file_path = "../../data/PEMS-SF/PEMS-SF_TRAIN.ts" # PEMS-SF 
    # file_path = "../../data/EthanolConcentration/EthanolConcentration_TRAIN.ts" # EthanolConcentration
    # file_path = "../../data/SelfRegulationSCP2/SelfRegulationSCP2_TRAIN.ts" # SCP2
    # file_path = "../../data/Heartbeat/Heartbeat_TRAIN.ts" # Heartbeat        
    # file_path = "../../data/Car/Car_TRAIN.ts" # Car
    # file_path = "../../data/Cricket/Cricket_TRAIN.ts" # Cricket
    file_path = "../../data/Coffee/Coffee_TRAIN.ts" # Coffee
        
    #Test files
    # file_path = "../../data/UWaveGestureLibrary/UWaveGestureLibrary_TEST.ts" # UWaveGestureLibrary
    # file_path = "../../data/JapaneseVowels/JapaneseVowels_eq_TEST.ts" # JapaneseVowels
    # file_path = "../../data/Handwriting/Handwriting_TEST.ts"
    # file_path = "../../data/BasicMotions/BasicMotions_TEST.ts"
    # file_path = "../../data/Cricket/Cricket_TEST.ts"
    # file_path = "../../data/Epilepsy/Epilepsy_TEST.ts"
    # file_path = "../../data/AtrialFibrillation/AtrialFibrillation_TEST.ts" # AtrialFibrillation
    # file_path = "../../data/SelfRegulationSCP2/SelfRegulationSCP2_TEST.ts" # SCP2
    
    
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

    L_min = int(0.2 * T)  # 20% length of T
    L_max = int(0.7 * T)  # 70% length of T
    L_step = max(5, int(0.1 * T))  # step size 10% of T or 5 if T is small

    L_list = list(range(L_min, L_max + 1, L_step))
    print(f"\n Auto Generated Shapelet Lengths: {L_list}\n")

    num_per_length = 40
    
    shapelets = generate_multivariate_shapelets(X, y, L_list, num_per_length)

    df_scores = evaluate_multivariate_shapelets(shapelets, X, y)

    df_topk = save_topk_balanced(
        shapelets, df_scores, "shapelets_Coffee.csv",
        num_classes=len(np.unique(y)), per_class=10
        
        # # 1 dim
        # shapelets, df_scores, "shapelets_Car.csv",
        # num_classes=len(np.unique(y)), per_class=10
        
        # shapelets, df_scores, "shapelets_UWaveGestureLibrary_test.csv",
        # num_classes=len(np.unique(y)), per_class=40
    )
