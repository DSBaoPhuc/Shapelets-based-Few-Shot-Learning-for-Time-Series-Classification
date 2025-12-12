# ------------------------------------------------------
# Information Gain Visualization for Shapelet Splitting

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import entropy

# # Cấu hình style cho đẹp (Seaborn style)
# sns.set_style("whitegrid")
# plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# def calculate_entropy(labels):
#     """Tính Shannon Entropy cho một danh sách nhãn"""
#     if len(labels) == 0:
#         return 0.0
#     values, counts = np.unique(labels, return_counts=True)
#     probs = counts / len(labels)
#     # Dùng log cơ số 2 cho đúng chuẩn Information Theory
#     return entropy(probs, base=2) 

# def visualize_ig_concept():
#     # ---------------------------------------------------------
#     # 1. Tạo dữ liệu giả lập (Simulation Data)
#     # ---------------------------------------------------------
#     np.random.seed(42)
    
#     # Giả sử đây là khoảng cách từ Shapelet đến các samples
#     # Class 0 (Xanh): Khoảng cách gần (mean=2.0)
#     # Class 1 (Đỏ): Khoảng cách xa (mean=6.0)
#     # Có một chút overlap ở giữa (khoảng 3.5 - 4.5) để bài toán thực tế hơn
#     dist_class_0 = np.random.normal(loc=2.5, scale=0.8, size=50)
#     dist_class_1 = np.random.normal(loc=5.5, scale=1.0, size=50)
    
#     distances = np.concatenate([dist_class_0, dist_class_1])
#     labels = np.concatenate([np.zeros(50), np.ones(50)])
    
#     # Sắp xếp dữ liệu theo khoảng cách
#     sorted_idx = np.argsort(distances)
#     distances = distances[sorted_idx]
#     labels = labels[sorted_idx]

#     # ---------------------------------------------------------
#     # 2. Tính toán Information Gain tại mọi điểm cắt
#     # ---------------------------------------------------------
#     parent_entropy = calculate_entropy(labels)
#     ig_values = []
#     thresholds = []
    
#     # Duyệt qua các điểm cắt khả thi (giữa 2 điểm dữ liệu liên tiếp)
#     for i in range(1, len(distances)):
#         # Threshold là trung điểm của 2 giá trị liên tiếp
#         thresh = (distances[i-1] + distances[i]) / 2
#         thresholds.append(thresh)
        
#         # Chia dữ liệu
#         left_labels = labels[:i]
#         right_labels = labels[i:]
        
#         # Tính Entropy con
#         w_left = len(left_labels) / len(labels)
#         w_right = len(right_labels) / len(labels)
#         child_entropy = w_left * calculate_entropy(left_labels) + \
#                         w_right * calculate_entropy(right_labels)
        
#         # Information Gain
#         ig = parent_entropy - child_entropy
#         ig_values.append(ig)

#     # Tìm điểm cắt tốt nhất (Max IG)
#     best_idx = np.argmax(ig_values)
#     best_threshold = thresholds[best_idx]
#     max_ig = ig_values[best_idx]

#     # Tính lại entropy tại điểm cắt tốt nhất để hiển thị
#     split_idx = best_idx + 1 # +1 vì thresholds có len = len(dist) - 1
#     left_ent = calculate_entropy(labels[:split_idx])
#     right_ent = calculate_entropy(labels[split_idx:])

#     # ---------------------------------------------------------
#     # 3. Vẽ biểu đồ (Visualization)
#     # ---------------------------------------------------------
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, 
#                                    gridspec_kw={'height_ratios': [1, 1]})
    
#     # --- Biểu đồ trên: Đường cong Information Gain ---
#     ax1.plot(thresholds, ig_values, color='#333333', linewidth=2, label='Information Gain Curve')
#     ax1.axvline(x=best_threshold, color='green', linestyle='--', alpha=0.8)
#     ax1.scatter([best_threshold], [max_ig], color='green', s=100, zorder=5)
    
#     # Annotate Max IG
#     ax1.annotate(f'Max IG = {max_ig:.3f}\nThreshold = {best_threshold:.2f}', 
#                  xy=(best_threshold, max_ig), xytext=(best_threshold + 1, max_ig),
#                  arrowprops=dict(facecolor='black', shrink=0.05),
#                  fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.9))
    
#     ax1.set_ylabel('Information Gain (Bits)', fontsize=12)
#     ax1.set_title('Step 1: Finding Optimal Split Threshold', fontsize=14, fontweight='bold')
#     ax1.legend(loc='upper right')

#     # --- Biểu đồ dưới: Phân bố dữ liệu và điểm cắt ---
#     # Vẽ các điểm dữ liệu (có jitter y để không bị chồng lên nhau)
#     y_jitter = np.random.normal(0, 0.05, size=len(distances))
    
#     # Class 0 points
#     ax2.scatter(distances[labels==0], y_jitter[labels==0], 
#                 color='blue', alpha=0.6, s=50, label='Class 0 (Close Matches)')
#     # Class 1 points
#     ax2.scatter(distances[labels==1], y_jitter[labels==1], 
#                 color='red', alpha=0.6, s=50, label='Class 1 (Non-Matches)')
    
#     # Vẽ đường cắt
#     ax2.axvline(x=best_threshold, color='green', linestyle='--', linewidth=2)
    
#     # Tô màu nền để phân biệt 2 vùng
#     ax2.axvspan(distances.min()-0.5, best_threshold, alpha=0.1, color='blue')
#     ax2.axvspan(best_threshold, distances.max()+0.5, alpha=0.1, color='red')
    
#     # Hiển thị Entropy của từng vùng
#     ax2.text(best_threshold - 1.5, 0.2, f'LEFT CHILD\nEntropy = {left_ent:.3f}\n(Pure Class 0)', 
#              ha='center', va='center', color='blue', fontweight='bold', 
#              bbox=dict(boxstyle="round", fc="white", ec="blue", alpha=0.8))
    
#     ax2.text(best_threshold + 1.5, 0.2, f'RIGHT CHILD\nEntropy = {right_ent:.3f}\n(Pure Class 1)', 
#              ha='center', va='center', color='red', fontweight='bold',
#              bbox=dict(boxstyle="round", fc="white", ec="red", alpha=0.8))

#     ax2.set_xlabel('Distance to Shapelet', fontsize=12)
#     ax2.set_ylabel('Samples (Jittered)', fontsize=12)
#     ax2.set_title(f'Step 2: Resulting Split', 
#                   fontsize=14, fontweight='bold')
#     ax2.set_yticks([]) # Ẩn trục Y vì nó chỉ là jitter
#     ax2.legend(loc='lower right')

#     plt.tight_layout()
#     plt.savefig('information_gain_visualization.png', dpi=300) # Lưu ảnh
#     plt.show()

# if __name__ == "__main__":
#     visualize_ig_concept()


# ------------------------------------------------------
# Separability Visualization for Shapelets

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Cấu hình giao diện chuẩn Thesis
# sns.set_style("whitegrid")
# plt.rcParams.update({
#     'font.size': 12, 
#     'font.family': 'serif',
#     'axes.titlesize': 14,
#     'axes.labelsize': 12
# })

# def plot_separability_scenario(ax, mean1, mean2, std1, std2, title, is_good_case=True):
#     """
#     Hàm vẽ một trường hợp phân bố khoảng cách cụ thể
#     """
#     # 1. Tạo dữ liệu giả lập
#     np.random.seed(42)
#     n_samples = 60
    
#     # Tạo cụm dữ liệu cho Class A (Xanh) và Class B (Đỏ)
#     dists_a = np.random.normal(mean1, std1, n_samples)
#     dists_b = np.random.normal(mean2, std2, n_samples)
    
#     # Tạo jitter cho trục Y để các điểm không đè lên nhau (giúp dễ nhìn hơn)
#     y_a = np.random.normal(1, 0.04, n_samples)
#     y_b = np.random.normal(1, 0.04, n_samples)
    
#     # 2. Vẽ các điểm dữ liệu (Scatter Plot)
#     ax.scatter(dists_a, y_a, color='blue', alpha=0.6, s=50, label='Class A Samples')
#     ax.scatter(dists_b, y_b, color='red', alpha=0.6, s=50, label='Class B Samples')
    
#     # 3. Vẽ trọng tâm (Centroids / Means)
#     mu_a = np.mean(dists_a)
#     mu_b = np.mean(dists_b)
    
#     ax.axvline(mu_a, color='blue', linestyle='--', linewidth=2, alpha=0.8)
#     ax.axvline(mu_b, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
#     # Ghi chú thích cho trọng tâm
#     ax.text(mu_a, 1.15, f'$\mu_A$', ha='center', va='bottom', color='blue', fontweight='bold', fontsize=14)
#     ax.text(mu_b, 1.15, f'$\mu_B$', ha='center', va='bottom', color='red', fontweight='bold', fontsize=14)
    
#     # 4. Vẽ mũi tên thể hiện Separability (Margin)
#     # Gap = Khoảng cách giữa 2 trọng tâm
#     mid_point_y = 1.1
#     ax.annotate(
#         text='', 
#         xy=(mu_a, mid_point_y), 
#         xytext=(mu_b, mid_point_y), 
#         arrowprops=dict(arrowstyle='<->', color='black', linewidth=2, shrinkA=0, shrinkB=0)
#     )
    
#     # Tính giá trị Gap
#     gap = abs(mu_b - mu_a)
    
#     # Hiển thị text giá trị Gap
#     text_color = 'green' if is_good_case else 'orange'
#     label_text = f"Separability (Gap) = {gap:.2f}"
    
#     ax.text((mu_a + mu_b)/2, mid_point_y + 0.02, label_text, 
#             ha='center', va='bottom', color='black', fontweight='bold',
#             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=text_color, linewidth=2))

#     # Tinh chỉnh trục
#     ax.set_yticks([]) # Ẩn trục Y vì nó chỉ là jitter
#     ax.set_ylim(0.8, 1.3)
#     ax.set_xlabel('Distance Value')
#     ax.set_title(title, fontweight='bold')
#     if is_good_case:
#         ax.legend(loc='lower right')

# def visualize_separability():
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
#     # --- Trường hợp 1: Separability Thấp (Bad Shapelet) ---
#     # Hai cụm gần nhau (Mean 2.0 và 3.5), độ lệch chuẩn lớn
#     plot_separability_scenario(
#         ax1, 
#         mean1=2.0, mean2=3.5, 
#         std1=0.6, std2=0.6, 
#         title="(A) Low Separability: High risk of misclassification",
#         is_good_case=False
#     )
    
#     # --- Trường hợp 2: Separability Cao (Good Shapelet) ---
#     # Hai cụm xa nhau (Mean 2.0 và 8.0), độ lệch chuẩn nhỏ
#     plot_separability_scenario(
#         ax2, 
#         mean1=2.0, mean2=8.0, 
#         std1=0.4, std2=0.4, 
#         title="(B) High Separability: Safe decision boundary",
#         is_good_case=True
#     )
    
#     plt.tight_layout()
#     plt.savefig('separability_visualization.png', dpi=300)
#     plt.show()

# if __name__ == "__main__":
#     visualize_separability()

# ------------------------------------------------------
# F-statistic Visualization for Shapelet
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# Cấu hình giao diện chuẩn Thesis (Font chữ có chân cho giống LaTeX)
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12, 
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

def plot_f_stat_scenario(ax, data_a, data_b, title_prefix):
    """
    Hàm vẽ phân phối và trực quan hóa các thành phần của F-stat
    """
    # 1. Tính toán F-score thực tế
    f_stat, p_val = f_oneway(data_a, data_b)
    
    # 2. Vẽ biểu đồ mật độ (KDE) để thấy rõ "Within-Variance" (độ bè của chuông)
    sns.kdeplot(data_a, ax=ax, fill=True, color='blue', alpha=0.2, linewidth=2, label='Class A Distances')
    sns.kdeplot(data_b, ax=ax, fill=True, color='red', alpha=0.2, linewidth=2, label='Class B Distances')
    
    # 3. Vẽ các điểm dữ liệu thực tế (Rug plot / Scatter) dưới đáy
    # Thêm jitter nhẹ trục y để nhìn rõ mật độ điểm
    y_a = np.random.normal(-0.05, 0.01, size=len(data_a))
    y_b = np.random.normal(-0.05, 0.01, size=len(data_b))
    ax.scatter(data_a, y_a, color='blue', s=20, alpha=0.6, marker='|')
    ax.scatter(data_b, y_b, color='red', s=20, alpha=0.6, marker='|')
    
    # 4. Tính toán Mean
    mean_a = np.mean(data_a)
    mean_b = np.mean(data_b)
    global_mean = np.mean(np.concatenate([data_a, data_b]))
    
    # Vẽ đường trung bình
    ax.axvline(mean_a, color='blue', linestyle='--', alpha=0.8)
    ax.axvline(mean_b, color='red', linestyle='--', alpha=0.8)
    
    # 5. Annotation: BETWEEN-GROUP VARIANCE (Khoảng cách giữa các mean)
    # Vẽ mũi tên nối 2 mean
    height_arrow = 0.25
    ax.annotate(
        text='', xy=(mean_a, height_arrow), xytext=(mean_b, height_arrow),
        arrowprops=dict(arrowstyle='<->', color='black', linewidth=2)
    )
    ax.text((mean_a + mean_b)/2, height_arrow + 0.02, 'Between-Group Var', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 6. Annotation: WITHIN-GROUP VARIANCE (Độ rộng của phân phối)
    # Vẽ mũi tên chỉ độ rộng của Class A
    ax.annotate(
        text='', xy=(mean_a - np.std(data_a), 0.1), xytext=(mean_a + np.std(data_a), 0.1),
        arrowprops=dict(arrowstyle='<->', color='blue', linewidth=1.5)
    )
    ax.text(mean_a, 0.12, 'Within-Var', ha='center', va='bottom', color='blue', fontsize=9)

    # Tinh chỉnh biểu đồ
    ax.set_title(f"{title_prefix}\nCalculated F-Statistic = {f_stat:.2f}", fontweight='bold')
    ax.set_xlabel("Distance Value")
    ax.set_ylabel("Density")
    ax.set_yticks([]) # Ẩn trục Y vì giá trị mật độ không quan trọng bằng hình dáng
    
    # Thêm text giải thích High/Low
    status_text = "GOOD Shapelet" if f_stat > 100 else "POOR Shapelet"
    status_color = "green" if f_stat > 100 else "orange"
    ax.text(0.02, 0.95, status_text, transform=ax.transAxes, 
            color=status_color, fontweight='bold', fontsize=12,
            bbox=dict(boxstyle="round", fc="white", ec=status_color, alpha=0.9))

def visualize_f_statistic():
    np.random.seed(10)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Kịch bản 1: F-Stat CAO (Tốt) ---
    # Within-Var nhỏ (std=0.4), Between-Var lớn (kc=6)
    d_high_a = np.random.normal(2, 0.4, 50)
    d_high_b = np.random.normal(8, 0.4, 50)
    
    plot_f_stat_scenario(ax1, d_high_a, d_high_b, "(A) High F-Statistic Scenario")
    ax1.legend(loc='upper right')

    # --- Kịch bản 2: F-Stat THẤP (Xấu) ---
    # Within-Var lớn (std=2.0), Between-Var nhỏ (kc=2) -> Chồng lấn
    d_low_a = np.random.normal(4, 2.0, 50)
    d_low_b = np.random.normal(6, 2.0, 50)
    
    plot_f_stat_scenario(ax2, d_low_a, d_low_b, "(B) Low F-Statistic Scenario")
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('f_statistic_visualization.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    visualize_f_statistic()