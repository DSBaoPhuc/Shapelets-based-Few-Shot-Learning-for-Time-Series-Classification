import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_excel("../pruned_covered_Gun_Point.xlsx")
# file_path = "../candidate_shapelets_Gun_Point.xlsx"
df2 = pd.read_excel("../_shapelets_Gun_Point.xlsx")


df1_plot = df1[['quality']].copy()
df1_plot['source'] = 'Pruned & Covered'

df2_plot = df2[['quality']].copy()
df2_plot['source'] = 'Covered Only'

# Combine both dataframes
plot_df = pd.concat([df1_plot, df2_plot], ignore_index=True)

plt.figure(figsize=(10, 6))
sns.kdeplot(data=plot_df, x='quality', hue='source', fill=True, common_norm=False, alpha=0.5)
plt.title('Distribution of Shapelet Quality Scores')
plt.xlabel('Quality')
plt.ylabel('Density')
plt.grid(True)
plt.tight_layout()
plt.show()
