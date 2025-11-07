import numpy as np
import pandas as pd
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# ==========================
# TS Loader (UEA .ts format)
# ==========================
def load_ts_file(file_path):
    with open(file_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    data_start = lines.index("@data") + 1
    X, y = [], []
    for line in lines[data_start:]:
        parts = line.split(":")
        dims = [np.array(list(map(float, d.split(","))), dtype=np.float32)
                for d in parts[:-1]]
        label = parts[-1]
        X.append(dims)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = LabelEncoder().fit_transform(y)
    return X, y


# ==========================
# CNN Encoder Few-Shot
# ==========================
class CNNEncoder(nn.Module):
    def __init__(self, in_channels, emb_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 9, padding=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 7, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, 5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


# ==========================
# Episode Creation
# ==========================
def create_episode(X, y, N=3, K=1, Q=1):
    unique_classes = np.unique(y)
    N = min(N, len(unique_classes))
    classes = np.random.choice(unique_classes, N, replace=False)

    sup_x, sup_y, qry_x, qry_y = [], [], [], []
    for i, c in enumerate(classes):
        idx = np.where(y == c)[0]
        chosen = np.random.choice(idx, K + Q, replace=False)
        sup_x.append(X[chosen[:K]])
        qry_x.append(X[chosen[K:]])
        sup_y.append(np.full(K, i))
        qry_y.append(np.full(Q, i))

    sup_x = torch.tensor(np.concatenate(sup_x), dtype=torch.float32)
    qry_x = torch.tensor(np.concatenate(qry_x), dtype=torch.float32)
    sup_y = torch.tensor(np.concatenate(sup_y), dtype=torch.long)
    qry_y = torch.tensor(np.concatenate(qry_y), dtype=torch.long)
    return sup_x, sup_y, qry_x, qry_y


# ==========================
# Proto Training Step
# ==========================
def proto_train_step(model, opt, X, y, device):
    model.train()
    sup_x, sup_y, qry_x, qry_y = create_episode(X, y)
    sup_x, qry_x = sup_x.to(device), qry_x.to(device)
    sup_y, qry_y = sup_y.to(device), qry_y.to(device)

    sup_emb = model(sup_x)
    qry_emb = model(qry_x)

    N = len(torch.unique(sup_y))
    prototypes = [sup_emb[sup_y == c].mean(0) for c in range(N)]
    prototypes = torch.stack(prototypes)

    d = torch.cdist(qry_emb, prototypes)
    loss = F.cross_entropy(-d, qry_y)

    opt.zero_grad()
    loss.backward()
    opt.step()
    acc = (d.argmin(1) == qry_y).float().mean().item()
    return loss.item(), acc


# ==========================
# Training
# ==========================
def train_model(X, y, episodes=200):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNEncoder(in_channels=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(episodes):
        loss, acc = proto_train_step(model, opt, X, y, device)
        if ep % 20 == 0:
            print(f"Episode {ep}: Loss={loss:.4f}, Acc={acc:.4f}")
    return model


# ==========================
# Proto Evaluation on TEST
# ==========================
def proto_evaluate(model, X_test, y_test, X_train, y_train):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        train_emb = model(torch.tensor(X_train, dtype=torch.float32).to(device))
        test_emb = model(torch.tensor(X_test, dtype=torch.float32).to(device))

    preds = []
    for te in test_emb:
        d = torch.norm(train_emb - te.unsqueeze(0), dim=1)
        pred = y_train[d.argmin().item()]
        preds.append(pred)

    acc = accuracy_score(y_test, preds)
    print("\nTEST Accuracy:", round(acc * 100, 4), "%")
    print(classification_report(y_test, preds))
    return acc


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":

    # ===== LOAD SHAPELET CSV =====
    csv_path = "Shapelet_extract/top_shapelets_BM_100%.csv"
    df = pd.read_csv(csv_path)

    df["Shapelet_Values"] = df["Shapelet_Values"].apply(ast.literal_eval)

    max_len = max(len(v) for v in df["Shapelet_Values"])
    padded = np.array([
        np.pad(v, (0, max_len - len(v)), "constant")
        for v in df["Shapelet_Values"]
    ], dtype=np.float32)

    y_train_shape = df["true_class"].values
    X_train_shape = padded[:, np.newaxis, :]

    print("Train Shapelets:", X_train_shape.shape, y_train_shape.shape)

    # ===== LOAD REAL TEST SET UEA =====
    X_test, y_test = load_ts_file("../../data/BasicMotions/BasicMotions_TEST.ts")
    print("Test set:", X_test.shape, y_test.shape)

    # ===== TRAIN FEW SHOT MODEL on shapelet dataset =====
    model = train_model(X_train_shape, y_train_shape)

    # ===== EVALUATE ON REAL TEST =====
    proto_evaluate(model, X_test, y_test, X_train_shape, y_train_shape)
