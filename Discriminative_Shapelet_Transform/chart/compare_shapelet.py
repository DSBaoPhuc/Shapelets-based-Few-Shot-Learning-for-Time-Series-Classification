import pandas as pd
import matplotlib.pyplot as plt

# file_path = '../result(IG)/covered_shapelets.xlsx'
# file_path = '../result(IG)/candidate_shapelets.xlsx' 
file_path = "../candidate_shapelets_Gun_Point.xlsx" 
data = pd.read_excel(file_path)

data = data.rename(columns=lambda x: x.strip())

# sorted_data = data.sort_values(by='values', ascending=False)
sorted_data = data.sort_values(by='quality', ascending=False)

top_shapelets = sorted_data.head(10)

plt.figure(figsize=(10, 6))
for index, row in top_shapelets.iterrows():
    shapelet = row['values'] 
    if isinstance(shapelet, str):
        shapelet = eval(shapelet)
    plt.plot(shapelet, label=f"Shapelet {index} (IG: {row['quality']:.2f})")

plt.title("Top 5 Similar Shapelets")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()
