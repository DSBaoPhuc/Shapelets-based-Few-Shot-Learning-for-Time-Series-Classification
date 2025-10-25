import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # để matplotlib biết sử dụng 3D projection
import numpy as np

# === Đọc dữ liệu từ file CSV ===
csv_path = "top_shapelets_multi.csv"   # đổi path nếu cần
df = pd.read_csv(csv_path)

# === Kiểm tra các cột cần thiết ===
required_cols = ["id", "IG", "F_stat", "Separability"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Thiếu các cột bắt buộc trong CSV: {missing}")

# === Tính Composite Score nếu chưa có ===
if "Composite_Score" not in df.columns:
    df["Composite_Score"] = (
        0.4 * df["IG"] + 
        0.3 * np.log1p(df["F_stat"]) + 
        0.3 * df["Separability"]
    )

# === Chuẩn hóa màu theo Composite_Score ===
colors = plt.cm.viridis(
    (df["Composite_Score"] - df["Composite_Score"].min()) / 
    (df["Composite_Score"].max() - df["Composite_Score"].min())
)

# === Tạo biểu đồ 3D ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    df["IG"], df["F_stat"], df["Separability"],
    c=colors, s=100, edgecolors='k', alpha=0.8
)

# === Ghi nhãn từng điểm shapelet ===
for _, row in df.iterrows():
    ax.text(row["IG"], row["F_stat"], row["Separability"],
            f'ID {int(row["id"])}', fontsize=9, color='black')

# === Nhãn và tiêu đề ===
ax.set_xlabel("Information Gain (IG)", fontsize=12, labelpad=10)
ax.set_ylabel("F-statistic", fontsize=12, labelpad=10)
ax.set_zlabel("Separability", fontsize=12, labelpad=10)
ax.set_title("3D Visualization of Top Shapelets by Quality Metrics", fontsize=14, pad=20)

# === Thanh màu thể hiện Composite Score ===
cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
cbar.set_label("Composite Score", fontsize=12)

plt.tight_layout()
plt.show()
