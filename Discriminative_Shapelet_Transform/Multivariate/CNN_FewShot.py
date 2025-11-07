import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
import ast


# ================================================
# Load shapelet CSV
# ================================================
def load_shapelet_csv(path):
    df = pd.read_csv(path)

    # Convert to numpy array
    X = [np.array(ast.literal_eval(v)) for v in df["Shapelet_Values"]]
    y = df["true_class"].values.astype(np.int64)

    return X, y


# ================================================
# Pad sequences
# ================================================
def pad_sequences(X, max_len=None):
    if max_len is None:
        max_len = max(len(v) for v in X)
    padded = []
    for v in X:
        diff = max_len - len(v)
        if diff < 0:
            v = v[:max_len]
            diff = 0
        padded.append(np.pad(v, (0, diff)))
    return np.array(padded), max_len


# ================================================
# CNN Encoder
# ================================================
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, emb_dim=64):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, emb_dim, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x.view(x.size(0), -1)


# ================================================
# Create Few-Shot Episode
# ================================================
def create_episode(X, y, N=4, K=2, Q=2, max_retries=20):
    unique_classes = np.unique(y)
    N = min(N, len(unique_classes))

    for _ in range(max_retries):
        classes = np.random.choice(unique_classes, N, replace=False)

        sup_x, sup_y, qry_x, qry_y = [], [], [], []
        valid = True

        for i, c in enumerate(classes):
            idx = np.where(y == c)[0]
            if len(idx) < K + Q:
                valid = False
                break
            chosen = np.random.choice(idx, K + Q, replace=False)
            sup_x.append(X[chosen[:K]])
            qry_x.append(X[chosen[K:]])
            sup_y.append(np.full(K, i))
            qry_y.append(np.full(Q, i))

        if valid:
            return (
                torch.tensor(np.concatenate(sup_x), dtype=torch.float32),
                torch.tensor(np.concatenate(sup_y), dtype=torch.long),
                torch.tensor(np.concatenate(qry_x), dtype=torch.float32),
                torch.tensor(np.concatenate(qry_y), dtype=torch.long)
            )

    raise RuntimeError("Few-shot sampling failed!")


# ================================================
# Train episode
# ================================================
def proto_train_step(model, opt, X, y, device, N=4, K=2, Q=2):
    model.train()
    sup_x, sup_y, qry_x, qry_y = create_episode(X, y, N, K, Q)
    sup_x, qry_x = sup_x.to(device), qry_x.to(device)
    sup_y, qry_y = sup_y.to(device), qry_y.to(device)

    sup_e = model(sup_x)
    qry_e = model(qry_x)

    prototypes = []
    for c in torch.unique(sup_y):
        prototypes.append(sup_e[sup_y == c].mean(dim=0))
    prototypes = torch.stack(prototypes)  # [N, emb]

    dist = torch.cdist(qry_e, prototypes)
    loss = F.cross_entropy(-dist, qry_y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    preds = torch.argmin(dist, dim=1)
    acc = (preds == qry_y).float().mean().item()

    return loss.item(), acc


# ================================================
# Train Model
# ================================================
def train_model(X, y, episodes=200, N=4, K=2, Q=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNEncoder(in_channels=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(episodes):
        loss, acc = proto_train_step(model, opt, X, y, device, N, K, Q)
        if ep % 20 == 0:
            print(f"Episode {ep}: Loss={loss:.4f}, Acc={acc:.4f}")

    return model


# ================================================
# Test
# ================================================
def test_model(model, X_train, y_train, X_test, y_test):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    emb_train = model(X_train)
    emb_test = model(X_test)

    preds = []
    for e in emb_test:
        d = torch.cdist(e.unsqueeze(0), emb_train)
        preds.append(y_train[torch.argmin(d)])

    preds = np.array(preds)
    print(f"\nTEST Accuracy: {100*np.mean(preds == y_test):.1f}%")
    print(classification_report(y_test, preds))


# ================================================
# MAIN
# ================================================
if __name__ == "__main__":
    train_path = "Shapelet_extract/top_shapelets_AF_40%.csv"
    test_path = "Shapelet_extract/Shapelet_Test/top_shapelets_AF_test.csv"

    X_train, y_train = load_shapelet_csv(train_path)
    X_test, y_test = load_shapelet_csv(test_path)

    # Fix sequences length
    X_train, max_len = pad_sequences(X_train)
    X_test, _ = pad_sequences(X_test, max_len=max_len)

    # Reshape for CNN
    X_train = X_train[:, None, :]
    X_test = X_test[:, None, :]

    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)

    # Few-Shot params
    N = 3  # 4-way
    K = 3  # 2-shot
    Q = 2  # 2-query

    model = train_model(X_train, y_train,
                        episodes=250,
                        N=N, K=K, Q=Q)

    test_model(model,
               X_train, y_train,
               X_test, y_test)
