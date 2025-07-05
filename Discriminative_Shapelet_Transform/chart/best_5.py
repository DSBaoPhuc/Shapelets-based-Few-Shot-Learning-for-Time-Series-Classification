import pandas as pd
import ast
import matplotlib.pyplot as plt

# file_path = "../candidate_shapelets_Gun_Point.xlsx"
file_path = "../result_IG(Gun_Point)/pruned_covered.xlsx"
# file_path = "../result(IG)/covered_shapelets.xlsx"
df = pd.read_excel(file_path)

# Chuyển đổi cột 'values' từ chuỗi sang danh sách
df['values'] = df['values'].apply(ast.literal_eval)

# Sắp xếp dữ liệu theo cột 'quality' tăng dần (vì quality âm, giá trị nhỏ hơn là tốt hơn)
df_sorted = df.sort_values(by='quality').head(5)

plt.figure(figsize=(10, 6))

for index, row in df_sorted.iterrows():
    shapelet_id = f"source_id={row['source_id']}, start_pos={row['start_pos']}"
    plt.plot(row['values'], label=shapelet_id)

plt.title("Top 5 Shapelets Based on Quality", fontsize=14)
plt.xlabel("Index", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.legend(title="Shapelets", fontsize=10)
plt.grid(alpha=0.5)
plt.show()