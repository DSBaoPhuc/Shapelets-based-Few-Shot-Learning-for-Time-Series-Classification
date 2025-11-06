import ast
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# =============================
# LOAD + NORMALIZE DATA
# =============================
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

def load_normalized(filename):
    X, y = load_ts_file(filename)
    n_samples, n_dims, _ = X.shape
    scalers = []

    for d in range(n_dims):
        scaler = StandardScaler()
        X[:, d, :] = scaler.fit_transform(X[:, d, :])
        scalers.append(scaler)

    return X, y, scalers

def normalize_test(X_test, scalers):
    n_dims = X_test.shape[1]
    for d in range(n_dims):
        X_test[:, d, :] = scalers[d].transform(X_test[:, d, :])
    return X_test

# =============================
# DISTANCE FUNCTION (min-dist)
# =============================
def compute_min_dist(shapelet, ts_2d):
    n_dims, sl_len = shapelet.shape
    _, ts_len = ts_2d.shape

    best = np.inf
    for i in range(ts_len - sl_len + 1):
        window = ts_2d[:, i:i+sl_len]
        dist = np.linalg.norm(window - shapelet)
        if dist < best:
            best = dist
    return best


def shapelet_transform(X, shapelets):
    n_samples = X.shape[0]
    n_shapelets = len(shapelets)
    feat = np.zeros((n_samples, n_shapelets), dtype=np.float32)

    for i in range(n_samples):
        for j, sl in enumerate(shapelets):
            feat[i, j] = compute_min_dist(sl, X[i])
    return feat

# Load TRAIN (normalize)
# X_train, y_train, scalers = load_normalized("../../data/BasicMotions/BasicMotions_TRAIN.ts") # BasicMotions
# X_train, y_train, scalers = load_normalized("../../data/StandWalkJump/StandWalkJump_TRAIN.ts") # StandWalkJump
# X_train, y_train, scalers = load_normalized("../../data/Libras/Libras_TRAIN.ts") # Libras
# X_train, y_train, scalers = load_normalized("../../data/RacketSports/RacketSports_TRAIN.ts") # RacketSports
X_train, y_train, scalers = load_normalized("../../data/Cricket/Cricket_TRAIN.ts") # Cricket
# X_train, y_train, scalers = load_normalized("../../data/Epilepsy/Epilepsy_TRAIN.ts") # Epilepsy
# X_train, y_train, scalers = load_normalized("../../data/ArticularyWordRecognition/ArticularyWordRecognition_TRAIN.ts") # ArticularyWordRecognition
# X_train, y_train, scalers = load_normalized("../../data/AtrialFibrillation/AtrialFibrillation_TRAIN.ts") # AtrialFibrillation
# X_train, y_train, scalers = load_normalized("../../data/FingerMovements/FingerMovements_TRAIN.ts") # FingerMovements
# X_train, y_train, scalers = load_normalized("../../data/Heartbeat/Heartbeat_TRAIN.ts") # Heartbeat
# X_train, y_train, scalers = load_normalized("../../data/NATOPS/NATOPS_TRAIN.ts") # NATOPS
# X_train, y_train, scalers = load_normalized("../../data/LSST/LSST_TRAIN.ts") # LSST
# X_train, y_train, scalers = load_normalized("../../data/SelfRegulationSCP1/SelfRegulationSCP1_TRAIN.ts") # SelfRegulationSCP1
# X_train, y_train, scalers = load_normalized("../../data/PenDigits/PenDigits_TRAIN.ts") # PenDigits


# Load TEST + normalize using TRAIN Scalers
# X_test, y_test = load_ts_file("../../data/BasicMotions/BasicMotions_TEST.ts") # BasicMotions
# X_test, y_test = load_ts_file("../../data/StandWalkJump/StandWalkJump_TEST.ts") # StandWalkJump
# X_test, y_test = load_ts_file("../../data/Libras/Libras_TEST.ts") # Libras
# X_test, y_test = load_ts_file("../../data/RacketSports/RacketSports_TEST.ts") # RacketSports
X_test, y_test = load_ts_file("../../data/Cricket/Cricket_TEST.ts") # Cricket
# X_test, y_test = load_ts_file("../../data/Epilepsy/Epilepsy_TEST.ts") # Epilepsy
# X_test, y_test = load_ts_file("../../data/ArticularyWordRecognition/ArticularyWordRecognition_TEST.ts") # ArticularyWordRecognition
# X_test, y_test = load_ts_file("../../data/AtrialFibrillation/AtrialFibrillation_TEST.ts") # AtrialFibrillation
# X_test, y_test = load_ts_file("../../data/FingerMovements/FingerMovements_TEST.ts") # FingerMovements
# X_test, y_test = load_ts_file("../../data/Heartbeat/Heartbeat_TEST.ts") # Heartbeat
# X_test, y_test = load_ts_file("../../data/NATOPS/NATOPS_TEST.ts") # NATOPS
# X_test, y_test = load_ts_file("../../data/LSST/LSST_TEST.ts") # LSST
# X_test, y_test = load_ts_file("../../data/SelfRegulationSCP1/SelfRegulationSCP1_TEST.ts") # SelfRegulationSCP1
# X_test, y_test = load_ts_file("../../data/PenDigits/PenDigits_TEST.ts") # PenDigits

X_test = normalize_test(X_test, scalers)

# Load shapelets from CSV
df = pd.read_csv("Shapelet_extract/top_shapelets_CK_93%.csv")

# Parse shapelet values
shapelets = []
for (_, row) in df.iterrows():
    sl_flat = ast.literal_eval(row["Shapelet_Values"])
    L = row["length"]
    shapelet = np.array(sl_flat).reshape(-1, L)  # Reshape to (n_dims, L)
    shapelets.append(shapelet)

print("Loaded", len(shapelets), "shapelets")

# Shapelet Transform
Xtr = shapelet_transform(X_train, shapelets)
Xte = shapelet_transform(X_test, shapelets)

print(" Feature shape TRAIN:", Xtr.shape)
print(" Feature shape TEST :", Xte.shape)

# Train classifier (SVM)
clf = SVC(kernel="rbf")
clf.fit(Xtr, y_train)

# Predict TEST
y_pred = clf.predict(Xte)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print("\n TEST Accuracy:", round(acc*100, 4), "%")
print("\n Classification Report:\n", classification_report(y_test, y_pred))

# Save model
model_path = "save_model/svm_shapelet_model_CK_93%.joblib"
joblib.dump(clf, model_path)

print(f"\n Model saved to: {model_path}")
