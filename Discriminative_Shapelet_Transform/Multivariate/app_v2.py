import streamlit as st
import os
import pandas as pd
import numpy as np
from PIL import Image
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# ==========================================
# CONFIG
# ==========================================
DIR_COMPOSITE = "Composite_Score"
DIR_SHAPELETS_IMG = "Shapelets_imgs"
DIR_TRAINING = "Training_results"
DIR_CSV = "Shapelet_extract"
DIR_MODELS = "saved_models"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_ACCURACY_MAP = {
    "BasicMotions": 98.8,
    "EthanolConcentration": 64.7,
    "FaceDetection": 73.2,
    "Handwriting": 53.6,
    "JapaneseVowels": 83.5,
    "PEMS-SF": 55.7,
    "SelfRegulationSCP1": 90.4,
    "SelfRegulationSCP2": 78.2,
    "SpokenArabicDigits": 70.8,
    "UWaveGestureLibrary": 95.53
}

# --- BILINGUAL DATASET DESCRIPTIONS ---
DATASET_DESCRIPTIONS = {
    "BasicMotions": {
        "vi": "Dữ liệu từ cảm biến gia tốc (smartwatch) ghi lại 4 hoạt động: Đi bộ, Chạy, Cầu lông và Đứng yên.",
        "en": "Accelerometer data (smartwatch) recording 4 activities: Walking, Running, Badminton, and Standing."
    },
    "EthanolConcentration": {
        "vi": "Dữ liệu quang phổ dùng để xác định nồng độ Ethanol trong các dung dịch hỗn hợp nước/rượu.",
        "en": "Spectral data used to determine Ethanol concentration in water/alcohol mixtures."
    },
    "FaceDetection": {
        "vi": "Dữ liệu MEG (tín hiệu não) phân loại phản ứng của não bộ khi nhìn thấy khuôn mặt so với các vật thể khác.",
        "en": "MEG data (brain signals) classifying brain responses to faces versus other objects."
    },
    "Handwriting": {
        "vi": "Dữ liệu tọa độ bút (x, y) ghi lại quá trình viết tay, mục tiêu là nhận diện chữ cái dựa trên chuyển động.",
        "en": "Pen tip coordinates (x, y) recording handwritten characters, aiming to recognize letters based on movement."
    },
    "JapaneseVowels": {
        "vi": "Dữ liệu âm thanh ghi lại phát âm nguyên âm tiếng Nhật của 9 người nói. Mục tiêu là nhận diện người nói.",
        "en": "Audio data recording Japanese vowel pronunciations from 9 speakers. Goal is speaker recognition."
    },
    "PEMS-SF": {
        "vi": "Dữ liệu giao thông (tỷ lệ chiếm dụng làn đường) từ các cảm biến cao tốc tại San Francisco Bay Area.",
        "en": "Traffic data (lane occupancy rates) collected from sensors on highways in the San Francisco Bay Area."
    },
    "SelfRegulationSCP1": {
        "vi": "Dữ liệu BCI ghi lại điện thế vỏ não chậm (SCP) khi người dùng cố gắng điều khiển con trỏ nhân tạo.",
        "en": "BCI data recording slow cortical potentials (SCP) when users try to control an artificial cursor."
    },
    "SelfRegulationSCP2": {
        "vi": "Tương tự SCP1 nhưng là tập dữ liệu mở rộng/khác biệt về điều kiện thu thập tín hiệu não bộ.",
        "en": "Similar to SCP1 but an extended/different dataset regarding brain signal collection conditions."
    },
    "SpokenArabicDigits": {
        "vi": "Dữ liệu MFCCs từ các đoạn ghi âm 10 chữ số Ả Rập (0-9) được nói bởi 88 người.",
        "en": "Mel-frequency cepstral coefficients (MFCCs) from recordings of 10 Arabic digits (0-9) spoken by 88 speakers."
    },
    "UWaveGestureLibrary": {
        "vi": "Dữ liệu gia tốc kế từ Wii Remote, ghi lại 8 mẫu cử chỉ tay đơn giản để điều khiển thiết bị.",
        "en": "Accelerometer data from a Wii Remote, recording 8 simple hand gesture patterns to control devices."
    }
}

# --- [UPDATED] STATIC COMPARISON TABLE DATA ---
STATIC_COMPARISON_DATA = {
    "Dataset": [
        "BasicMotions", "EthanolConcentration", "FaceDetection", "Handwriting", 
        "JapaneseVowels", "PEMS-SF", "SelfRegulationSCP1", "SelfRegulationSCP2", "SpokenArabicDigits", "UWaveGestureLibrary"
    ],
    # dataset information (Dimensions, Length, Class, Train/Test Size)
    "Train Size": [40, 261, 5890, 150, 270, 267, 268, 200, 6599, 120],
    "Test Size":  [40, 263, 3524, 850, 370, 173, 293, 180, 2199, 320],
    "Dimensions": [6, 3, 144, 3, 12, 963, 6, 7, 13, 3],
    "Length":     [100, 1751, 62, 152, 29, 144, 896, 1152, 93, 315],
    "Classes":    [4, 4, 2, 26, 9, 7, 2, 2, 10, 8],
    # Accuracy
    "Paper Accuracy [1] (%)": [100, 52.1, 68.6, 36.4, 78.7, 49.6, 88.5, 75.2, 66.7, 78.2],
    "Avg Accuracy (%)":       [98.8, 64.7, 73.2, 53.6, 83.5, 55.7, 90.4, 78.2, 70.8, 95.53]
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_available_datasets():
    if not os.path.exists(DIR_COMPOSITE):
        return []
    files = os.listdir(DIR_COMPOSITE)
    datasets = [f.replace(".png", "") for f in files if f.endswith(".png")]
    return sorted(datasets)

def load_image(image_path):
    if os.path.exists(image_path):
        return Image.open(image_path)
    return None

def find_csv_for_dataset(dataset_name):
    if not os.path.exists(DIR_CSV):
        return None
    pattern = os.path.join(DIR_CSV, f"*shapelets_{dataset_name}*.csv")
    files = glob.glob(pattern)
    if files:
        return files[0]
    return None

# ==========================================
# MODEL DEFINITION (COPIED FROM TRAINING)
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, pool=False):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2) if pool else None

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        if self.pool:
            x = self.pool(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)

class CNNEncoder(nn.Module):
    def __init__(self, in_channels, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=7, padding=3, pool=True),
            ConvBlock(32, 32, kernel_size=5, padding=2, pool=True),
            ResidualBlock(32),
            ConvBlock(32, 64, kernel_size=3, padding=1, pool=True),
            ResidualBlock(64),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return F.normalize(x, dim=1)

# ==========================================
# LOAD MODEL (FIXED)
# ==========================================
@st.cache_resource
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)

    model = CNNEncoder(
        in_channels=checkpoint["in_channels"],
        emb_dim=checkpoint["emb_dim"]
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    return model

def pad_shapelet(sh, T_max):
    """
    sh: np.array [D, T]
    return: [D, T_max]
    """
    D, T = sh.shape
    if T < T_max:
        pad_width = ((0, 0), (0, T_max - T))
        sh = np.pad(sh, pad_width, mode="constant")
    else:
        sh = sh[:, :T_max]
    return sh



# ==========================================
# LOAD SHAPELETS (MULTIVARIATE)
# ==========================================
def load_shapelets_multidim(csv_path):
    df = pd.read_csv(csv_path)

    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    D = len(dim_cols)

    T_max = df["Shapelet_Length"].max()

    X, y = [], []

    for _, row in df.iterrows():
        dims = []
        for d in dim_cols:
            dims.append(np.array(eval(row[d])))

        sh = np.stack(dims)        # [D, T]
        sh = pad_shapelet(sh, T_max)

        X.append(sh)
        y.append(row["true_class"])

    X = np.stack(X)   # [B, D, T_max]
    y = np.array(y)

    return X, y


def pad_to_maxlen(X, max_len):
    B, D, T = X.shape
    if T < max_len:
        X = np.pad(X, ((0,0),(0,0),(0,max_len-T)))
    elif T > max_len:
        X = X[:, :, :max_len]
    return X

def compute_prototypes(embeddings, labels):
    """
    embeddings: [N, emb_dim]
    labels: [N]
    return: dict {class_id: prototype_embedding}
    """
    prototypes = {}
    for c in np.unique(labels):
        proto = embeddings[labels == c].mean(dim=0)
        prototypes[int(c)] = proto
    return prototypes


def classify_shapelets(embeddings, prototypes):
    """
    embeddings: [N, emb_dim]
    prototypes: dict {class_id: embedding}
    return: list predicted classes
    """
    proto_keys = list(prototypes.keys())
    proto_stack = torch.stack([prototypes[c] for c in proto_keys])  # [C, emb_dim]

    dists = torch.cdist(embeddings, proto_stack)  # [N, C]
    preds = torch.argmin(dists, dim=1)

    return [proto_keys[i] for i in preds.cpu().numpy()]




# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Analysis Dashboard", layout="wide")

st.title("Shapelets-based Few-Shot Learning for Time Series Classification Analysis")
st.markdown("---")

# Sidebar
datasets = sorted([f.replace(".png","") for f in os.listdir(DIR_COMPOSITE)])
selected_dataset = st.sidebar.selectbox("Dataset", datasets)

# model_files = glob.glob(os.path.join(DIR_MODELS, "*.pt"))
# model_path = st.sidebar.selectbox("Model", model_files)

# model = load_model(model_path)
# st.sidebar.success("Model loaded correctly")

# --- 2. METRIC CARDS ---
best_acc = DATASET_ACCURACY_MAP.get(selected_dataset, "N/A")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Dataset Name", value=selected_dataset)
with col2:
    st.metric(label="Best Accuracy", value=f"{best_acc}%" if best_acc != "N/A" else "N/A")
with col3:
    st.metric(label="Total Shapelets per Class", value="5")

# --- SHOW BILINGUAL DESCRIPTION ---
desc_data = DATASET_DESCRIPTIONS.get(selected_dataset, {"vi": "Chưa có mô tả.", "en": "No description available."})
st.info(f"""
**Data Description:**
* {desc_data['en']}
* {desc_data['vi']}
""")

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Overview",
    "📈 Training",
    "🔮 Predict"
])

# tab1, tab2, tab3, tab4 = st.tabs([
#     "📊 Overview",
#     "📋 Shapelets",
#     "📈 Training",
#     "🔮 Predict"
# ])

# === TAB 1: OVERVIEW & SHAPELETS ===
with tab1:
    st.subheader(f"Best Shapelet per Class - {selected_dataset}")
    csv_file = find_csv_for_dataset(selected_dataset)
    
    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            if "Composite_Score" in df.columns and "true_class" in df.columns:
                df_sorted = df.sort_values(by="Composite_Score", ascending=False)
                df_best = df_sorted.groupby("true_class").head(1).sort_values("true_class").reset_index(drop=True)
                
                display_cols = ['id', 'true_class', 'length', 'F_stat', "Separability", 'IG', 'Composite_Score']
                final_cols = [c for c in display_cols if c in df_best.columns]
                
                st.dataframe(df_best[final_cols], use_container_width=True)
                
                csv_data = df_best[final_cols].to_csv(index=False).encode('utf-8')
                st.download_button("Download Filtered Data", data=csv_data, file_name=f"best_shapelets_{selected_dataset}.csv", mime="text/csv")
            else:
                st.error("CSV file missing 'Composite_Score' or 'true_class' columns.")
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        st.warning(f"CSV file for **{selected_dataset}** not found in `{DIR_CSV}`.")
    
    col_left, col_right = st.columns([1.5, 1])
    
    with col_right:
        st.subheader("Composite Score Analysis")
        comp_img_path = os.path.join(DIR_COMPOSITE, f"{selected_dataset}.png")
        img_comp = load_image(comp_img_path)
        if img_comp:
            st.image(img_comp, caption="3D Shapelet Evaluation Metrics", use_container_width=True)
        else:
            st.warning("Image not found.")

    with col_left:
        st.subheader("Best Shapelets Visualization")
        ds_img_dir = os.path.join(DIR_SHAPELETS_IMG, selected_dataset)
        
        if os.path.exists(ds_img_dir):
            image_files = sorted(glob.glob(os.path.join(ds_img_dir, "*.png")))
        else:
             st.warning(f"Looking in root folder {DIR_SHAPELETS_IMG} (Sub-folder recommended).")
             image_files = sorted(glob.glob(os.path.join(DIR_SHAPELETS_IMG, "Class_*.png")))

        if image_files:
            grid_cols = st.columns(2)
            for idx, img_p in enumerate(image_files):
                img = load_image(img_p)
                with grid_cols[idx % 2]:
                    st.image(img, caption=os.path.basename(img_p), use_container_width=True)
        else:
            st.info("No shapelet visualization images found.")

# # === TAB 2: DETAILED DATA ===
# with tab2:
#     st.subheader(f"Best Shapelet per Class - {selected_dataset}")
#     csv_file = find_csv_for_dataset(selected_dataset)
    
#     if csv_file:
#         try:
#             df = pd.read_csv(csv_file)
#             if "Composite_Score" in df.columns and "true_class" in df.columns:
#                 df_sorted = df.sort_values(by="Composite_Score", ascending=False)
#                 df_best = df_sorted.groupby("true_class").head(1).sort_values("true_class").reset_index(drop=True)
                
#                 display_cols = ['id', 'true_class', 'length', 'F_stat', "Separability", 'IG', 'Composite_Score']
#                 final_cols = [c for c in display_cols if c in df_best.columns]
                
#                 st.dataframe(df_best[final_cols], use_container_width=True)
                
#                 csv_data = df_best[final_cols].to_csv(index=False).encode('utf-8')
#                 st.download_button("Download Filtered Data", data=csv_data, file_name=f"best_shapelets_{selected_dataset}.csv", mime="text/csv")
#             else:
#                 st.error("CSV file missing 'Composite_Score' or 'true_class' columns.")
#                 st.dataframe(df.head())
#         except Exception as e:
#             st.error(f"Error reading CSV: {e}")
#     else:
#         st.warning(f"CSV file for **{selected_dataset}** not found in `{DIR_CSV}`.")

# === TAB 2: TRAINING PROCESS ===
with tab2:
    st.subheader("Training Loss & Accuracy")
    train_img_path = os.path.join(DIR_TRAINING, f"training_loss_acc_{selected_dataset}.png")
    img_train = load_image(train_img_path)
    if img_train:
        st.image(img_train, use_container_width=True)
    else:
        st.warning(f"Training log image not found: {train_img_path}")

st.markdown("---")

# TAB 3: PREDICTION
with tab3:
    st.subheader("Predict Shapelet Classes (Prototype-based)")
    
    model_files = glob.glob(os.path.join(DIR_MODELS, "*.pt"))
    model_path = st.selectbox("Model", model_files)

    model = load_model(model_path)
    # st.sidebar.success("Model loaded correctly")

    uploaded = st.file_uploader("Upload shapelet CSV", type=["csv"])
    if uploaded:
        X_test, y_test = load_shapelets_multidim(uploaded)

        X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        y_test = torch.tensor(y_test)

        with torch.no_grad():
            emb = model(X_test)

        # ---- Prototype inference ----
        prototypes = compute_prototypes(emb, y_test)
        pred_classes = classify_shapelets(emb, prototypes)

        # ---- Display result ----
        result_df = pd.DataFrame({
            "True_Class": y_test.cpu().numpy(),
            "Predicted_Class": pred_classes
        })

        st.success("Shapelet classification completed")
        st.dataframe(result_df, use_container_width=True)

        acc = (result_df["True_Class"] == result_df["Predicted_Class"]).mean()
        st.metric("Shapelet Classification Accuracy", f"{acc:.4f}")


st.caption("MS-ProtoNet | Thesis Dashboard")
