import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# === Load CSV shapelet top-k ===
csv_path = "top_shapelets_balanced_SWJ.csv"
df = pd.read_csv(csv_path)

# === Check required columns ===
required_cols = ["id", "true_class", "pred_class", "start", "length",
                 "F_stat", "Separability", "IG", "Composite_Score"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Miss columns: {missing}")

# === Normalize class mapping and colors ===
unique_classes = sorted(df["pred_class"].unique())
color_map = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
class_to_color = {cls: color_map[i] for i, cls in enumerate(unique_classes)}

# === Colormap based on Composite Score ===
norm_scores = (df["Composite_Score"] - df["Composite_Score"].min()) / \
               (df["Composite_Score"].max() - df["Composite_Score"].min())
cmap = plt.cm.viridis(norm_scores)

# === Plot 3D ===
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    df["Separability"], df["F_stat"], df["IG"],
    c=cmap, s=150,
    edgecolors=[class_to_color[c] for c in df["pred_class"]],
    linewidths=1.5,
    alpha=0.92
)

# === Annotation for each shapelet ===
for _, r in df.iterrows():
    text = f"ID:{int(r['id'])}\nT:{int(r['true_class'])}/P:{int(r['pred_class'])}\n@{int(r['start'])}x{int(r['length'])}"
    ax.text(
        r["Separability"], r["F_stat"], r["IG"],
        text, fontsize=7, alpha=0.85
    )

# === Axes Labels ===
ax.set_xlabel("Separability", fontsize=14, labelpad=12)
ax.set_ylabel("F-statistic", fontsize=14, labelpad=12)
ax.set_zlabel("Information Gain (IG)", fontsize=14, labelpad=12)
ax.set_title("Top-K Shapelets - Composite Score Colormap + IG as Z Axis", fontsize=16, pad=20)

# === Class Legend (edgecolors) ===
for cls in unique_classes:
    ax.scatter([], [], [], edgecolors=class_to_color[cls],
               facecolors="none", s=150, linewidths=2, label=f"Pred Class {cls}")

ax.legend(loc="upper left", fontsize=12)

# === Colormap Legend (Composite Score) ===
mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
mappable.set_array(df["Composite_Score"])
cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Composite Score", fontsize=12)

plt.tight_layout()
plt.show()
