import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
import json
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from copy import deepcopy
import random
import matplotlib.pyplot as plt

# -----------------------------
# Utility: load / pad
# -----------------------------
def load_shapelet_csv_multidim(path):
    df = pd.read_csv(path)
    # detect dim columns
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    dim_cols = sorted(dim_cols, key=lambda x: int(x.split("_")[1]))
    
    X, y = [], []
    for _, row in df.iterrows():
        dims = []
        for col in dim_cols:
            val = row[col]
            if pd.isna(val):
                continue
            arr = json.loads(val) if isinstance(val, str) else val
            dims.append(np.array(arr, dtype=np.float32))
        if len(dims) == 0:
            continue
        # stack dims: shape (n_dims, L)
        shapelet_2d = np.stack(dims, axis=0)
        X.append(shapelet_2d)
        y.append(int(row["true_class"]))
    return X, np.array(y, dtype=np.int64)

def pad_shapelets(X, max_len=None):
    # X: list of shapelets, each (n_dims, L)
    if max_len is None:
        max_len = max(shapelet.shape[1] for shapelet in X)
    padded = []
    for shapelet in X:
        n_dims, L = shapelet.shape
        pad_width = max_len - L
        if pad_width > 0:
            shapelet = np.pad(shapelet, ((0, 0), (0, pad_width)))
        elif pad_width < 0:
            shapelet = shapelet[:, :max_len]
        padded.append(shapelet)
    return np.stack(padded, axis=0), max_len  # shape: [B, n_dims, T]

# -----------------------------
# Simple augmentations
# -----------------------------
def augment_jitter(x, sigma=0.02):
    return x + np.random.normal(0, sigma, size=x.shape)

def augment_scaling(x, sigma=0.1):
    factor = np.random.normal(1.0, sigma)
    return x * factor

def random_crop(x, keep_ratio=0.9):
    L = len(x)
    keep = max(1, int(L * keep_ratio))
    if keep == L:
        return x
    start = np.random.randint(0, L - keep + 1)
    cropped = x[start:start+keep]
    return np.pad(cropped, (0, L - keep))

def augment_array(arr, p=0.5):
    out = arr.copy()
    n_dims, T = out.shape[1], out.shape[2]  # (B, n_dims, T)
    for i in range(out.shape[0]):
        if random.random() < p:
            op = random.choice([augment_jitter, augment_scaling, random_crop])
            new_shapelet = np.zeros_like(out[i])
            for d in range(n_dims):
                tmp = op(out[i,d])
                # đảm bảo padding/truncation để giữ đúng T
                L = len(tmp)
                if L < T:
                    tmp = np.pad(tmp, (0, T-L))
                elif L > T:
                    tmp = tmp[:T]
                new_shapelet[d] = tmp
            out[i] = new_shapelet
    return out

# -----------------------------
# CNN Encoder
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, pool=False):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2) if pool else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.pool is not None:
            x = self.pool(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, emb_dim=64, hidden_channels=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, kernel_size=7, padding=3, pool=True),
            ConvBlock(hidden_channels, hidden_channels, kernel_size=5, padding=2, pool=True),
            ResidualBlock(hidden_channels),
            ConvBlock(hidden_channels, hidden_channels*2, kernel_size=3, padding=1, pool=True),
            ResidualBlock(hidden_channels*2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels*2, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        self._init_weights()

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def temperature(self):
        return torch.clamp(self.log_temperature.exp(), min=1e-2, max=10.0)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

# -----------------------------
# Episode / Few-Shot utils
# -----------------------------
def create_episode(X, y, N=2, K=2, Q=3, max_retries=20):
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

def create_episode_batch(X, y, N=2, K=2, Q=3, batch_size=4):
    sup_xs, sup_ys, qry_xs, qry_ys = [], [], [], []
    for _ in range(batch_size):
        s_x, s_y, q_x, q_y = create_episode(X, y, N, K, Q)
        sup_xs.append(s_x)
        sup_ys.append(s_y)
        qry_xs.append(q_x)
        qry_ys.append(q_y)
    return (
        torch.cat(sup_xs, dim=0),
        torch.cat(sup_ys, dim=0),
        torch.cat(qry_xs, dim=0),
        torch.cat(qry_ys, dim=0)
    )

def proto_train_step(model, opt, X, y, device, N=2, K=2, Q=3, episodes_per_batch=4, clip_grad=1.0):
    model.train()
    sup_x, sup_y, qry_x, qry_y = create_episode_batch(X, y, N, K, Q, batch_size=episodes_per_batch)

    sup_x = augment_array(sup_x.numpy(), p=0.5)
    qry_x = augment_array(qry_x.numpy(), p=0.2)

    sup_x = torch.tensor(sup_x, dtype=torch.float32).to(device)
    qry_x = torch.tensor(qry_x, dtype=torch.float32).to(device)
    sup_y, qry_y = sup_y.to(device), qry_y.to(device)

    sup_e = model(sup_x)
    qry_e = model(qry_x)

    # compute prototypes
    prototypes = []
    for ei in range(episodes_per_batch):
        start = ei * (N*K)
        end = (ei+1)*(N*K)
        sup_block = sup_e[start:end]
        sup_y_block = sup_y[start:end]
        proto_block = []
        for c in torch.unique(sup_y_block):
            proto_block.append(sup_block[sup_y_block==c].mean(dim=0))
        prototypes.append(torch.stack(proto_block))
    prototypes = torch.cat(prototypes, dim=0)

    all_logits, all_targets = [], []
    for ei in range(episodes_per_batch):
        qstart = ei * (N*Q)
        qend = (ei+1)*(N*Q)
        qry_block = qry_e[qstart:qend]
        proto_block = prototypes[ei*N:(ei+1)*N]
        dist = torch.cdist(qry_block, proto_block, p=2)**2
        logits = -dist / model.temperature()
        all_logits.append(logits)
        all_targets.append(qry_y[qstart:qend])

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    loss = F.cross_entropy(logits, targets)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    opt.step()

    preds = torch.argmax(logits, dim=1)
    acc = (preds == targets).float().mean().item()
    return loss.item(), acc


def train_model(X, y, episodes=200, N=2, K=2, Q=3, episodes_per_batch=4,
                emb_dim=128, lr=3e-4, weight_decay=1e-4, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    in_channels = X.shape[1]
    model = CNNEncoder(in_channels=in_channels, emb_dim=emb_dim).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=max(1, episodes//10))

    best_model = deepcopy(model.state_dict())
    best_acc = -1.0

    history = {"loss": [], "acc": []}

    for ep in range(episodes):
        loss, acc = proto_train_step(model, opt, X, y, device, N, K, Q, episodes_per_batch)
        scheduler.step()

        history["loss"].append(loss)
        history["acc"].append(acc)

        if (ep+1) % 10 == 0 or ep == 0:
            print(f"Episode {ep+1:04d}/{episodes}: Loss={loss:.4f}, Acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model = deepcopy(model.state_dict())

    model.load_state_dict(best_model)
    return model, history

# -----------------------------
# Test k-NN on embeddings
# -----------------------------
def test_model(model, X_train, y_train, X_test, y_test):
    model.eval()
    device = next(model.parameters()).device
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test  = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        emb_train = model(X_train)
        emb_test  = model(X_test)

    preds = []
    for e in emb_test:
        d = torch.cdist(e.unsqueeze(0), emb_train, p=2)**2
        idx = torch.argmin(d, dim=1).item()
        preds.append(y_train[idx])
    preds = np.array(preds)
    print(f"\nTEST Accuracy: {100*np.mean(preds==y_test):.1f}%")
    print(classification_report(y_test, preds))

# -----------------------------
# Run full training & testing
# -----------------------------
if __name__ == "__main__":
    train_path = "Shapelet_extract/shapelets_JapaneseVowels.csv"
    test_path = "Shapelet_extract/Shapelet_Test/shapelets_JapaneseVowels_test.csv"
    
    # train_path = "Shapelet_extract/shapelets_UWaveGestureLibrary_95_train.csv"
    # # test_path = "shapelets_UWaveGestureLibrary.csv"
    # test_path = "Shapelet_extract/Shapelet_Test/shapelets_UWaveGestureLibrary_95_test.csv"

    X_train, y_train = load_shapelet_csv_multidim(train_path)
    X_test, y_test = load_shapelet_csv_multidim(test_path)

    X_train, max_len = pad_shapelets(X_train)
    X_test, _ = pad_shapelets(X_test, max_len=max_len)

    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)

    # Training
    model, history = train_model(
        X_train, y_train,
        episodes=2000,
        N=5, K=1, Q=4,
        # N=3, K=3, Q=3,
        
        episodes_per_batch=8,
        emb_dim=256,
        lr=3e-4,
        weight_decay=1e-4,
        device=None
    )

    # Plot
    losses = history["loss"]
    accs = history["acc"]
    epochs = np.arange(1, len(losses) + 1)

    plt.figure(figsize=(10,4))
    # subplot 1: loss
    plt.subplot(1,2,1)
    plt.plot(epochs, losses)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)

    # subplot 2: accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, accs)
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Test
    test_model(model, X_train, y_train, X_test, y_test)
