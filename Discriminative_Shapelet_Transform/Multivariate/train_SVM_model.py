import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.ensemble import RandomForestClassifier

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
    """
    shapelet: np.array shape (n_dims, sl_len)
    ts_2d:   np.array shape (n_dims, ts_len)
    """
    n_dims, sl_len = shapelet.shape
    _, ts_len = ts_2d.shape

    best = np.inf
    for i in range(ts_len - sl_len + 1):
        window = ts_2d[:, i:i+sl_len]
        # L2 on flattened multivariate window
        diff = window - shapelet
        dist = np.linalg.norm(diff)
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

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    # ---------- Load TRAIN (normalize) ----------
    X_train, y_train, scalers = load_normalized("../../data/EthanolConcentration/EthanolConcentration_TRAIN.ts") # Car

    # ---------- Load TEST + normalize ----------
    X_test, y_test = load_ts_file("../../data/EthanolConcentration/EthanolConcentration_TEST.ts")
    X_test = normalize_test(X_test, scalers)

    # ---------- Load shapelets from CSV (new format: dim_0, dim_1, ..., Shapelet_Length) ----------
    # csv_path = "Shapelet_extract/top_shapelets_BasicMotions.csv"
    csv_path = "Shapelet_extract/shapelets_EthanolConcentration_test.csv"
    df = pd.read_csv(csv_path)

    # Find all dim_* columns sorted by index
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    # sort by numeric suffix
    def dim_key(col):
        try:
            return int(col.split("_")[1])
        except:
            return col
    dim_cols = sorted(dim_cols, key=dim_key)

    shapelets = []
    for _, row in df.iterrows():
        # Collect available dimensions
        dims = []
        L = None
        for col in dim_cols:
            val = row.get(col, None)
            if pd.isna(val):
                # missing -> skip / treat as empty
                continue
            try:
                arr = json.loads(val) if isinstance(val, str) else val
            except Exception:
                # fallback: if it's already a python list representation without quotes
                try:
                    arr = eval(val)
                except Exception:
                    arr = []

            if isinstance(arr, list) and len(arr) > 0:
                arr_np = np.array(arr, dtype=np.float32)
                dims.append(arr_np)
                if L is None:
                    L = arr_np.shape[0]

        # If there's a Shapelet_Length column, we can cross-check
        if L is None and "Shapelet_Length" in df.columns:
            try:
                L = int(row["Shapelet_Length"])
            except Exception:
                L = None

        if len(dims) == 0:
            # skip empty row (shouldn't happen normally)
            continue

        # Ensure all dims have same length L (if not, pad / truncate to min length)
        lens = [d.shape[0] for d in dims]
        if len(set(lens)) != 1:
            # match to minimum length to be safe
            minL = min(lens)
            dims = [d[:minL] for d in dims]
            L = minL

        shapelet = np.vstack(dims)  # shape: (n_dims, L)
        shapelets.append(shapelet)

    print("Loaded", len(shapelets), "shapelets")

    # ---------- Shapelet Transform ----------
    Xtr = shapelet_transform(X_train, shapelets)
    Xte = shapelet_transform(X_test, shapelets)

    print(" Feature shape TRAIN:", Xtr.shape)
    print(" Feature shape TEST :", Xte.shape)

    # ---------- Train classifier (SVM) ----------
    clf = SVC(kernel="rbf")
    clf.fit(Xtr, y_train)

    # ---------- Train classifier (Random Forest) ----------
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(Xtr, y_train)

    # ---------- Predict TEST ----------
    y_pred = clf.predict(Xte)
    y_pred_rf = rf.predict(Xte)

    # ---------- Evaluate ----------
    acc = accuracy_score(y_test, y_pred)    
    print("\n Classification Report (SVM):\n", classification_report(y_test, y_pred))

    acc_rf = accuracy_score(y_test, y_pred_rf)
    print("\n Classification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))
    
    print("\nTEST Accuracy (SVM):", round(acc*100, 4), "%")
    print("TEST Accuracy (Random Forest):", round(acc_rf*100, 4), "%")

    # ---------- (optional) Save models ----------
    # joblib.dump(clf, "save_model/svm_shapelet_model.joblib")
    # joblib.dump(rf, "save_model/rf_shapelet_model.joblib")
