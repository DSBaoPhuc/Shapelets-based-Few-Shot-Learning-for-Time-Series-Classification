import streamlit as st
import os
import pandas as pd
from PIL import Image
import glob

# ==========================================
# File path constants
# ==========================================
DIR_COMPOSITE = "Composite_Score"
DIR_SHAPELETS_IMG = "Shapelets_imgs"
DIR_TRAINING = "Training_results"
DIR_CSV = "Shapelet_extract"

# Static accuracy data for Metric Card
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
# MAIN INTERFACE
# ==========================================
st.set_page_config(page_title="Analysis Dashboard", layout="wide")

st.title("Shapelets-based Few-Shot Learning for Time Series Classification Analysis")
st.markdown("---")

# --- 1. SIDEBAR ---
st.sidebar.header("Select Dataset")
datasets = get_available_datasets()

if not datasets:
    st.error(f"No image data found in `{DIR_COMPOSITE}`.")
    st.stop()

selected_dataset = st.sidebar.selectbox("Choose dataset:", datasets)
st.sidebar.markdown("---")
st.sidebar.info(f"Viewing: **{selected_dataset}**")

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

# --- 3. MAIN CONTENT TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Overview & Visuals", "📋 Detailed Data", "📈 Training Process"])

# === TAB 1: OVERVIEW & SHAPELETS ===
with tab1:
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

# === TAB 2: DETAILED DATA ===
with tab2:
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

# === TAB 3: TRAINING PROCESS ===
with tab3:
    st.subheader("Training Loss & Accuracy")
    train_img_path = os.path.join(DIR_TRAINING, f"training_loss_acc_{selected_dataset}.png")
    img_train = load_image(train_img_path)
    if img_train:
        st.image(img_train, use_container_width=True)
    else:
        st.warning(f"Training log image not found: {train_img_path}")

st.markdown("---")

# ==========================================
# 4. STATIC COMPARISON TABLE
# ==========================================
st.header("Performance Comparison Summary")
st.markdown("Compared to State-of-the-Art (SOTA) Results")

df_static = pd.DataFrame(STATIC_COMPARISON_DATA)

def highlight_selected(row):
    if row['Dataset'] == selected_dataset:
        return ['background-color: #d1e7dd; font-weight: bold; color: black'] * len(row)
    return [''] * len(row)

styled_df = df_static.style.format({
    "Paper Accuracy [1] (%)": "{:.2f}",
    "Avg Accuracy (%)": "{:.2f}"
}).apply(highlight_selected, axis=1)

st.table(styled_df)

# --- [ADD] IEEE CITATION ---
st.markdown("**Reference:**")
st.markdown("""
<div style="font-size: 14px; color: #555;">
[1] Y. Chen, Z. Li, C. Yang, X. Wang, and G. Xu, 
"Large Language Models are Few-shot Multivariate Time Series Classifiers," 
<i>arXiv preprint</i>, submitted Jan. 30, 2025.
</div>
""", unsafe_allow_html=True)