import numpy as np
import pandas as pd
import joblib
import os

# ==============================
# 1. Load saved models + imputer
# ==============================
save_dir = "./saved_models"

# Load imputer đã lưu lúc train
imputer_path = os.path.join(save_dir, "imputer.pkl")
if os.path.exists(imputer_path):
    imputer = joblib.load(imputer_path)
    print("Loaded imputer from imputer.pkl")
else:
    imputer = None
    print("No imputer found, predictions may fail if NaN exists")

# Load models
model_files = [f for f in os.listdir(save_dir) if f.endswith("_model.pkl")]
models = {}
for file in model_files:
    model_name = file.replace("_model.pkl", "")
    model_path = os.path.join(save_dir, file)
    models[model_name] = joblib.load(model_path)
    print(f"Loaded model: {model_name} from {file}")

# ==============================
# 2. New shapelets to predict
# ==============================
new_shapelets = [
    [0.014446999877691269, -0.6474800109863281, -0.2692300081253052, -0.20619000494480133, 0.6133300065994263, 1.3697999715805054],
    
    [0.5697699785232544, 0.19513000547885895, -0.08585599809885025, -0.17951999604701996, -0.2731800079345703, -0.08585599809885025],
    
    [1.0467000007629395, 0.6473199725151062, 0.5075399875640869, 0.6073799729347229, 0.6273499727249146, 0.6872599720954895]    
]

# Pad shapelets thành cùng chiều dài (12 feature)
max_len = imputer.n_features_in_  # = 12
X_new = pd.DataFrame([s + [np.nan] * (max_len - len(s)) for s in new_shapelets])

# Nếu có imputer -> transform trước khi predict
if imputer is not None:
    X_new = pd.DataFrame(imputer.transform(X_new), columns=range(max_len))

# ==============================
# 3. Predict using each model
# ==============================
for model_name, clf in models.items():
    preds = clf.predict(X_new)
    print("---------------------------------")
    print(f"Predictions by {model_name}:")
    for i, p in enumerate(preds):
        print(f"  Shapelet {i+1} -> Class {p}")
