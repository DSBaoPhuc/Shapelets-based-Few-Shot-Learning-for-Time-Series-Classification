import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import os

train_data = np.loadtxt("../../data/Gun_Point/Gun_Point_TRAIN", delimiter=',')  
X_train = train_data[:, 1:]  # Time series                              
y_train = train_data[:, 0]   # Labels

pruned_shapelets_df = pd.read_excel("../pruned_shapelets_Gun_Point.xlsx")
covered_shapelets_df = pd.read_excel("../pruned_covered_Gun_Point.xlsx")

def plot_shapelets(shapelet_df, title, num_plot=10, save_path=None):
    num_plot = min(num_plot, len(shapelet_df))  # control number of shapelets to plot
    plt.figure(figsize=(14, num_plot * 2.5))  

    for i in range(num_plot):
        row = shapelet_df.iloc[i]
        series_id = int(row['source_id'])
        start_pos = int(row['start_pos'])
        length = int(row['length'])
        class_label = int(row['class_label'])
        shapelet = np.array(ast.literal_eval(row['values']))

        series = X_train[series_id]
        label = int(y_train[class_label])

        plt.subplot(num_plot, 1, i+1)
        # plt.plot(series, label=f"Original Series (Label {label})", alpha=0.7)
        plt.plot(series, label=f"Original Series)", alpha=0.7)
        plt.plot(range(start_pos, start_pos + length), shapelet, color='red', linewidth=2.5, label="Shapelet")
        plt.title(f"Series ID: {series_id}, Shapelet Pos: {start_pos}, Length: {length}", fontsize=10)
        plt.legend(fontsize=8)

    plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # space for suptitle
    plt.subplots_adjust(hspace=0.8)  # increase space between subplot

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")

    plt.show()

plot_shapelets(pruned_shapelets_df, 
               title="Pruned Shapelets",
               num_plot=10,
               save_path="./figures/pruned_shapelets_GP.png")

plot_shapelets(covered_shapelets_df, 
               title="Covered Shapelets",
               num_plot=10,
               save_path="./figures/pruned_covered_GP.png")




