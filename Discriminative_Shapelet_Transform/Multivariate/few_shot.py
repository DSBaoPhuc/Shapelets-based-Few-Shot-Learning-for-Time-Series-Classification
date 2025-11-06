import numpy as np
import pandas as pd
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================
# CNN Encoder for Multivariate Time Series
# ======================================
class CNNEncoder(nn.Module):
    def __init__(self, in_channels, emb_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


# ======================================
# Episode Sampling for Few-Shot
# ======================================
def create_episode(X, y, N=5, K=1, Q=1, max_retries=10):
    unique_classes = np.unique(y)
    N = min(N, len(unique_classes))

    for _ in range(max_retries):
        classes = np.random.choice(unique_classes, N, replace=False)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for i, c in enumerate(classes):
            idx = np.where(y == c)[0]
            
            # if len(idx) < K + Q:
            if len(idx) < K + Q + 1:
                # lớp không đủ sample → thử lại episode mới
                break

            chosen = np.random.choice(idx, K + Q, replace=False)
            support_ids = chosen[:K]
            query_ids = chosen[K:]

            support_x.append(X[support_ids])
            query_x.append(X[query_ids])
            support_y.append(np.full(K, i))
            query_y.append(np.full(Q, i))

        # Kiểm tra nếu sampling hợp lệ
        if len(support_x) == N:
            support_x = torch.tensor(np.concatenate(support_x), dtype=torch.float32)
            support_y = torch.tensor(np.concatenate(support_y), dtype=torch.long)
            query_x = torch.tensor(np.concatenate(query_x), dtype=torch.float32)
            query_y = torch.tensor(np.concatenate(query_y), dtype=torch.long)
            return support_x, support_y, query_x, query_y

    raise ValueError("Cannot sample a valid episode - dataset too small for given N, K, Q")



# ======================================
# Few-Shot Training Step (Prototypical)
# ======================================
def proto_train_step(encoder, optimizer, X, y, device):
    encoder.train()
    # num_classes = len(np.unique(y))
    if min(np.bincount(y)) < 2:
        raise ValueError("Each class must have at least 2 samples for K=1 and Q=1.")
    
    N_way = min(3, len(np.unique(y)))
    # support_x, support_y, query_x, query_y = create_episode(X, y, N=num_classes, K=1, Q=1)
    support_x, support_y, query_x, query_y = create_episode(X, y, N=N_way, K=1, Q=1)

    support_x, query_x = support_x.to(device), query_x.to(device)
    support_y = support_y.to(device)
    query_y = query_y.to(device)

    sup_emb = encoder(support_x)
    qry_emb = encoder(query_x)

    N = len(torch.unique(support_y))
    prototypes = []
    for c in range(N):
        prototypes.append(sup_emb[support_y == c].mean(0))
    prototypes = torch.stack(prototypes)

    dists = torch.cdist(qry_emb, prototypes)
    loss = F.cross_entropy(-dists, query_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = (dists.argmin(dim=1) == query_y).float().mean().item()
    return loss.item(), acc


# ======================================
# Main Training Entry
# ======================================
def train_model(X, y, episodes=200, lr=1e-3, patience=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = CNNEncoder(in_channels=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    best_acc = 0.0
    no_improve = 0

    for ep in range(episodes):
        loss, acc = proto_train_step(encoder, optimizer, X, y, device)

        if acc > best_acc:
            best_acc = acc
            no_improve = 0
        else:
            no_improve += 1

        if ep % 10 == 0:
            print(f"[Episode {ep}] Loss={loss:.4f}, Acc={acc:.4f}, Best={best_acc:.4f}")

        # Early stopping
        if no_improve >= patience:
            print(f"Early stopping triggered at Episode {ep}")
            break

    print(f"Training completed! Best Accuracy = {best_acc:.4f}")
    return encoder


# ======================================
# Load CSV Shapelet Dataset & Train Few-Shot
# ======================================
if __name__ == "__main__":

    csv_path = "Shapelet_extract/top_shapelets_AF_40%.csv"
    df = pd.read_csv(csv_path)

    # Convert string → numpy array
    df["Shapelet_Values"] = df["Shapelet_Values"].apply(
        lambda x: np.array(ast.literal_eval(x), dtype=np.float32)
    )

    # Find max length and padding
    max_len = max(len(v) for v in df["Shapelet_Values"])
    padded = np.array([
        np.pad(v, (0, max_len - len(v)), "constant")
        for v in df["Shapelet_Values"]
    ])

    y = df["true_class"].values

    # Reshape: (samples, channels, seq_len)
    X = padded[:, np.newaxis, :]

    print("Data loaded:", X.shape, y.shape)
    print("Classes:", np.unique(y))

    model = train_model(X, y)
    print("Training completed!")
