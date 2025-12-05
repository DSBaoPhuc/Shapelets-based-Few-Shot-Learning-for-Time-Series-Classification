import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import ast
import os  # <--- [ADD] Import os để xử lý file/folder

# ------------------------------------------------------
# Load dataset (.ts multivariate)
# ------------------------------------------------------
def load_ts_file(file_path):
    print(f"Loading dataset from: {file_path} ...")
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    try:
        data_start = lines.index("@data") + 1
    except ValueError:
        data_start = 0 

    X, y = [], []
    for line in lines[data_start:]:
        parts = line.split(":")
        # Parse từng dimension
        dims = [np.array(list(map(float, dim.split(","))), dtype=np.float32)
                for dim in parts[:-1]]
        label = parts[-1].strip()
        X.append(dims)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = LabelEncoder().fit_transform(y)
    return X, y


# ------------------------------------------------------
# Parse shapelet from CSV
# ------------------------------------------------------
def load_shapelet_from_row(row):
    dim_cols = [col for col in row.index if col.startswith("dim_")]
    dim_cols = sorted(dim_cols, key=lambda x: int(x.split("_")[1]))

    dims = []
    for col in dim_cols:
        arr = ast.literal_eval(row[col])
        dims.append(np.array(arr, dtype=np.float32))

    return np.stack(dims)

# ------------------------------------------------------
# Compute best matching position
# ------------------------------------------------------
def subdist(series, shapelet):
    D, T = series.shape
    D_s, L = shapelet.shape
    
    if D != D_s:
        raise ValueError(f"Dimension mismatch: Series {D}, Shapelet {D_s}")

    best_dist = np.inf
    best_pos = 0

    # Sliding window
    for s in range(T - L + 1):
        window = series[:, s:s+L]
        # Euclidean distance
        dist = np.linalg.norm(window - shapelet)
        if dist < best_dist:
            best_dist = dist
            best_pos = s

    return best_dist, best_pos


# ------------------------------------------------------
# Plot shapelet
# ------------------------------------------------------
def plot_shapelet_on_series(series, shapelet, best_pos, shapelet_info, save_dir=None):
    D, L = shapelet.shape
    sid = shapelet_info['series_id']
    cls = shapelet_info['true_class']
    f_stat = shapelet_info.get('F_stat', 0)
    score = shapelet_info.get('Composite_Score', 0)
    
    plt.figure(figsize=(10, 6))

    colors = plt.cm.get_cmap('tab10', D) # Tạo bảng màu
    for d in range(D):
        plt.plot(series[d], color=colors(d), alpha=0.3, label=f"Dim {d+1} (Original)")

    # Vẽ vùng Shapelet match (nét đậm) trên series gốc
    for d in range(D):
        plt.plot(
            np.arange(best_pos, best_pos + L),
            series[d, best_pos:best_pos + L],
            color=colors(d),
            linewidth=3,
            # label=f"Shapelet Dim {d} (Matched)"
        )

    title_str = f"Class {cls} | Series ID: {sid} | Start: {best_pos} | Score: {score:.2f}"
    plt.title(title_str)
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    plt.tight_layout()

    # save image
    if save_dir:
        # Đặt tên file: Class_X_Series_Y.png
        # filename = f"Class_{cls}_Series_{sid}_Pos_{best_pos}.png"
        filename = f"Class_{cls}.png"
        full_path = os.path.join(save_dir, filename)
        plt.savefig(full_path)
        print(f"   >>> Image saved to: {full_path}")
    # ------------------------
    plt.show()
    plt.close() # Đóng figure để giải phóng bộ nhớ


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
if __name__ == "__main__":

    # 1. Load dataset
    # file_path = "../../data/BasicMotions/BasicMotions_TRAIN.ts"
    file_path = "../../data/UWaveGestureLibrary/UWaveGestureLibrary_TRAIN.ts"
    try:
        X, y = load_ts_file(file_path)
        print(f"Dataset Shape: {X.shape}, Labels: {np.unique(y)}")
    except FileNotFoundError:
        print(f"Error: Không tìm thấy file tại {file_path}. Hãy kiểm tra lại đường dẫn.")
        exit()

    # 2. Load shapelets
    csv_path = "Shapelet_extract/shapelets_UWaveGestureLibrary.csv"
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} shapelets from CSV.")
    except FileNotFoundError:
        print(f"Error: Không tìm thấy file CSV tại {csv_path}.")
        exit()

    # --- [ADD] Create Save Directory ---
    save_folder = "Shapelets_imgs"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Created directory: {save_folder}")
    else:
        print(f"Directory exists: {save_folder}")
    # -----------------------------------

    # 3. Lọc shapelet tốt nhất cho mỗi class
    sort_col = "Composite_Score" if "Composite_Score" in df.columns else "F_stat"
    df_sorted = df.sort_values(by=sort_col, ascending=False)
    
    # Group by 'true_class' và lấy dòng đầu tiên (cao nhất)
    best_shapelets = df_sorted.groupby("true_class").head(1).reset_index(drop=True)
    
    print(f"\nFound {len(best_shapelets)} representative shapelets (1 per class). Plotting and Saving...\n")

    # 4. Loop qua từng shapelet tốt nhất và vẽ
    for idx, row in best_shapelets.iterrows():
        try:
            # Lấy thông tin
            target_class = row["true_class"]
            series_id = int(row["series_id"])  # ID của chuỗi gốc
            
            # Kiểm tra bounds
            if series_id >= len(X):
                print(f"Warning: Series ID {series_id} vượt quá kích thước dataset ({len(X)}). Bỏ qua.")
                continue

            # Lấy chuỗi gốc tương ứng từ tập dữ liệu
            original_series = X[series_id]
            
            # Parse shapelet data
            shapelet_data = load_shapelet_from_row(row)

            # Tìm vị trí khớp
            dist, best_pos = subdist(original_series, shapelet_data)
            
            print(f"Class {target_class}: Plotting shapelet from Series ID {series_id} at pos {best_pos}")

            plot_shapelet_on_series(
                original_series, 
                shapelet_data, 
                best_pos, 
                shapelet_info=row,
                save_dir=save_folder  
            )
            
        except Exception as e:
            print(f"Error plotting class {row['true_class']}: {e}")