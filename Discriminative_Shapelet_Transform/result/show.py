import pandas as pd
import ast
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file Excel
df = pd.read_excel("covered_shapelets.xlsx")  # thay bằng đường dẫn thực tế

# Chuyển cột 'values' từ chuỗi thành list Python
df['values'] = df['values'].apply(ast.literal_eval)

# Chọn 5 shapelet có quality cao nhất (ở đây giả sử quality càng lớn càng tốt)
# Nếu quality là độ lỗi thì cần sắp xếp tăng dần
top5 = df.sort_values(by='quality', ascending=False).head(5)

# Vẽ 5 shapelet trên cùng một đồ thị
plt.figure(figsize=(10, 6))
for i, row in top5.iterrows():
    plt.plot(row['values'], label=f"source_id={row['source_id']}, quality={row['quality']:.2f}")

plt.title("Top 5 Shapelets")
plt.xlabel("Index")
plt.ylabel("Quality")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
