import numpy as np
import pandas as pd
import os, joblib, ast
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from joblib import Parallel, delayed

# ====== Custom functions ======
from Discriminative_Shapelet_Transform.Univariate.utils_uni import z_norm, subdist
from Discriminative_Shapelet_Transform.Univariate.utils_uni import load_ucr_dataset_tensor  # Hàm bạn đang dùng để load dataset

# ======================= SHAPELET TRANSFORM =======================
def shapelet_transform(X, shapelets):
    """
    Tính feature matrix bằng cách đo khoảng cách min từ mỗi chuỗi đến từng shapelet.
    Trả về ma trận [n_samples, n_shapelets].
    """
    n_samples = len(X)
    n_shapelets = len(shapelets)
    results = np.zeros((n_samples, n_shapelets))

    # parallel processing
    def compute_distance(i, j):
        return subdist(shapelets[j], X[i])

    print(f"Computing shapelet transform: {n_samples} samples × {n_shapelets} shapelets...")
    results = Parallel(n_jobs=-1, prefer="threads")(delayed(compute_distance)(i, j)
                    for i in range(n_samples) for j in range(n_shapelets))
    
    return np.array(results).reshape(n_samples, n_shapelets)

# ======================= MAIN PIPELINE =======================
def main():
    # === Load dataset ===
    X_train_raw, y_train = load_ucr_dataset_tensor('../data/Gun_Point/Gun_Point_TRAIN')
    X_test_raw, y_test = load_ucr_dataset_tensor('../data/Gun_Point/Gun_Point_TEST')
    # X_train_raw, y_train = load_ucr_dataset_tensor('../data/Coffee/Coffee_TRAIN')
    # X_test_raw, y_test = load_ucr_dataset_tensor('../data/Coffee/Coffee_TEST')
    # X_train_raw, y_train = load_ucr_dataset_tensor('../data/ItalyPowerDemand/ItalyPowerDemand_TRAIN')
    # X_test_raw, y_test = load_ucr_dataset_tensor('../data/ItalyPowerDemand/ItalyPowerDemand_TEST')

    print(f"Train shape: {X_train_raw.shape}, Test shape: {X_test_raw.shape}")
    print(f"Train classes: {np.unique(y_train)}, Test classes: {np.unique(y_test)}")

    # === Load extracted shapelets ===
    # file_path = 'result/candidate_shapelets_ITL.csv'
    file_path = 'pruned_covered_GP.csv'
    data = pd.read_csv(file_path)
    data['values'] = data['values'].apply(ast.literal_eval)
    shapelets = [np.array(s) for s in data['values']]
    print(f"Loaded {len(shapelets)} shapelets from {file_path}")

    # === Shapelet transform ===
    X_train_features = shapelet_transform(X_train_raw, shapelets)
    X_test_features = shapelet_transform(X_test_raw, shapelets)

    # === Imputation (if needed) ===
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train_features)
    X_test_imputed = imputer.transform(X_test_features)
    os.makedirs("saved_models/GP", exist_ok=True)
    joblib.dump(imputer, "saved_models/GP/imputer.pkl")

    print(f"Shapelet feature train shape: {X_train_imputed.shape}")
    print(f"Shapelet feature test shape: {X_test_imputed.shape}")

    # === Train and evaluate classifiers ===
    classifiers = {
        # "1-NN": KNeighborsClassifier(n_neighbors=1),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=300),
        "SVM_RBF": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
        # "Logistic_Regression": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,50), activation="relu", solver="adam", max_iter=300, random_state=42),
        "Gradient_Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.005, random_state=42),
    }

    for name, clf in classifiers.items():
        print("=================================")
        print(f"Training {name} ...")
        clf.fit(X_train_imputed, y_train)
        y_pred = clf.predict(X_test_imputed)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"{name} Accuracy: {acc:.4f}")
        print("Confusion matrix:\n", cm)

        model_path = os.path.join("saved_models/ITL", f"{name.replace(' ', '_')}_model.pkl")
        joblib.dump(clf, model_path)
        print(f"Saved model to {model_path}\n")

# ======================= RUN =======================
if __name__ == "__main__":
    main()
