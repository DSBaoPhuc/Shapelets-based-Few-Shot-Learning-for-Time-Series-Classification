import numpy as np
import pandas as pd
import joblib
import os

# ==============================
# 1. Load saved models + imputer
# ==============================
# save_dir = "./saved_models/GP"
save_dir = "./saved_models/GP"

# Load saved imputer if exists 
# imputer_path = os.path.join(save_dir, "GP_imputer.pkl")
imputer_path = os.path.join(save_dir, "imputer.pkl")

if os.path.exists(imputer_path):
    imputer = joblib.load(imputer_path)
    print("Loaded imputer from imputer.pkl")
else:
    imputer = None
    print("No imputer found, predictions may fail if NaN exists")

# Load models
model_files = [f for f in os.listdir(save_dir) if f.endswith("_model.pkl")]
# model_files = [f for f in os.listdir(save_dir) if f.endswith(".pkl") and f != "imputer.pkl"]
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
    [0.9959200024604797, 0.9914100170135498, 0.989329993724823, 0.9886299967765808, 0.9908000230789185, 0.9898200035095215, 0.9880899786949158, 0.9932799935340881, 0.9932799935340881, 0.9895300269126892, 0.9943400025367737, 0.9873899817466736, 0.9946200251579285, 0.9901099801063538, 0.9982100129127502, 0.997189998626709, 0.991890013217926, 0.9948099851608276, 0.9973700046539307, 1.00409996509552, 1.0047999620437622, 0.999239981174469, 0.9961699843406677, 1.000100016593933, 0.9825699925422668, 0.9702200293540955, 0.9420700073242188, 0.8823999762535095, 0.8298199772834778, 0.7515100240707397, 0.6691799759864807, 0.5708699822425842, 0.4507400095462799, 0.3198600113391876],
    [-0.5450400114059448, -0.6778500080108643, -0.8004800081253052, -0.9133599996566772, -1.0055999755859375, -1.100600004196167, -1.1571999788284302, -1.1866999864578247, -1.1969000101089478, -1.197100043296814, -1.1914000511169434, -1.1806000471115112, -1.169800043106079, -1.1507999897003174],
    [-0.7744699716567993, -0.7741400003433228, -0.7699499726295471, -0.7689999938011169, -0.7694699764251709, -0.7711799740791321, -0.7703099846839905, -0.7610499858856201, -0.7191699743270874, -0.6638399958610535, -0.639270007610321, -0.6344199776649475, -0.6287000179290771],
    [-0.6747999787330627, -0.6275200247764587, -0.5912200212478638, -0.565060019493103, -0.5695899724960327, -0.5896700024604797, -0.6199700236320496, -0.6491699814796448, -0.656719982624054, -0.6374599933624268, -0.6291700005531311, -0.6084700226783752, -0.5905900001525879, -0.5731099843978882]

]
# Pad shapelets to the same length as training data
max_len = imputer.n_features_in_  # for example: 18 (GP)
X_new = pd.DataFrame([s + [np.nan] * (max_len - len(s)) for s in new_shapelets])

# If had imputer -> transform before predict
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
