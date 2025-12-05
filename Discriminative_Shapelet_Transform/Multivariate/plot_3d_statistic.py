import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os  # <--- [ADD] Import os để xử lý folder

# === Load CSV shapelet top-k ===
# csv_path = "Shapelet_extract/shapelets_BasicMotions.csv"
# csv_path = "Shapelet_extract/shapelets_EthanolConcentration.csv"
# csv_path = "Shapelet_extract/shapelets_JapaneseVowels.csv"
# csv_path = "Shapelet_extract/shapelets_SelfRegulationSCP2_1.csv"
csv_path = "Shapelet_extract/shapelets_UWaveGestureLibrary.csv"
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: Không tìm thấy file tại {csv_path}")
    exit()

# === Check required columns ===
required_cols = ["id", "true_class", "pred_class", "start", "length",
                 "F_stat", "Separability", "IG", "Composite_Score"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Miss columns: {missing}")

# === Normalize class mapping and colors ===
unique_classes = sorted(df["true_class"].unique())
color_map = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
class_to_color = {cls: color_map[i] for i, cls in enumerate(unique_classes)}

# === Colormap based on Composite Score ===
norm_scores = (df["Composite_Score"] - df["Composite_Score"].min()) / \
               (df["Composite_Score"].max() - df["Composite_Score"].min())
cmap = plt.cm.viridis(norm_scores)

# === Plot 3D ===
fig = plt.figure(figsize=(10, 6)) # Tăng kích thước hình một chút cho rõ
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    df["Separability"], df["F_stat"], df["IG"],
    c=cmap, s=150,
    edgecolors=[class_to_color[c] for c in df["pred_class"]],
    linewidths=1.5,
    alpha=0.92
)

# === Annotation for each shapelet ===
# for _, r in df.iterrows():
#     # text = f"ID:{int(r['id'])}\nT:{int(r['true_class'])}/P:{int(r['pred_class'])}\n@{int(r['start'])}x{int(r['length'])}"
#     # text = f"T:{int(r['true_class'])}/P:{int(r['pred_class'])}\nL:{int(r['length'])}"
#     ax.text(
#         r["Separability"], r["F_stat"], r["IG"],
#         # text, fontsize=7, alpha=0.85
#     )

# === Axes Labels ===
ax.set_xlabel("Separability", fontsize=14, labelpad=12)
ax.set_ylabel("F-statistic", fontsize=14, labelpad=12)
ax.set_zlabel("Information Gain (IG)", fontsize=14, labelpad=12)
ax.set_title("3D Shapelet Evaluation Metrics", fontsize=16, pad=20)

# === Class Legend (edgecolors) ===
for cls in unique_classes:
    ax.scatter([], [], [], edgecolors=class_to_color[cls],
               facecolors="none", s=150, linewidths=2, label=f"Class {cls}")

ax.legend(loc="upper left", fontsize=12)

# === Colormap Legend (Composite Score) ===
mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
mappable.set_array(df["Composite_Score"])
cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Composite Score", fontsize=12)

plt.tight_layout()

# === [ADD] SAVE IMAGE CODE ===
save_folder = "Composite_Score"

# 1. Tạo folder nếu chưa tồn tại
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"Created directory: {save_folder}")

# 2. Định nghĩa tên file output
output_filename = "3D_Shapelets_Analysis.png"
save_path = os.path.join(save_folder, output_filename)

# 3. Lưu hình (Lưu ý: Phải gọi savefig TRƯỚC show)
# plt.savefig(save_path, dpi=300, bbox_inches='tight')
# print(f"Image saved successfully to: {save_path}")

# ==============================

plt.show()