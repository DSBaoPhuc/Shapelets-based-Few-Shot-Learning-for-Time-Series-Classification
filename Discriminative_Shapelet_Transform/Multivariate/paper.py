# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.patches as patches

# def draw_composite_components():
#     # 1. Cấu hình Style (Thesis/Paper Standard)
#     sns.set_style("whitegrid")
#     plt.rcParams.update({
#         'font.family': 'serif',
#         'font.size': 12,
#         'axes.labelsize': 12,
#         'axes.titlesize': 14,
#         'xtick.labelsize': 10,
#         'ytick.labelsize': 10
#     })

#     # Tạo Figure 1 hàng 3 cột
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
#     # Màu sắc
#     c_cls0 = '#1f77b4' # Xanh (Class A)
#     c_cls1 = '#d62728' # Đỏ (Class B)
#     c_split = 'green'  # Màu đường cắt

#     # ==========================================
#     # PANEL A: F-STATISTIC (Cluster Tightness)
#     # ==========================================
#     ax0 = axes[0]
    
#     # Tạo dữ liệu: Phương sai nội bộ RẤT NHỎ (Tight)
#     np.random.seed(42)
#     d0_tight = np.random.normal(2, 0.3, 50)
#     d1_tight = np.random.normal(6, 0.3, 50)
    
#     # Vẽ KDE (Mật độ)
#     sns.kdeplot(d0_tight, ax=ax0, color=c_cls0, fill=True, alpha=0.3, label='Class A')
#     sns.kdeplot(d1_tight, ax=ax0, color=c_cls1, fill=True, alpha=0.3, label='Class B')
    
#     # Vẽ các mũi tên chỉ độ hẹp (Variance)
#     # Arrow cho Class A
#     ax0.annotate('', xy=(2-0.3, 0.6), xytext=(2+0.3, 0.6),
#                  arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
#     ax0.text(2, 0.7, r'Small $\sigma^2_{within}$', ha='center', fontsize=10, fontweight='bold')
    
#     # Arrow cho Class B
#     ax0.annotate('', xy=(6-0.3, 0.6), xytext=(6+0.3, 0.6),
#                  arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    
#     ax0.set_title('(A) F-Statistic\n(Cluster Tightness)', fontweight='bold')
#     ax0.set_xlabel('Distance Value')
#     ax0.set_yticks([])
#     ax0.legend(loc='upper right')
    
#     # Chú thích thêm
#     ax0.text(0.05, 0.9, "High F-Score means\ntight clusters", transform=ax0.transAxes, 
#              fontsize=10, style='italic', bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

#     # ==========================================
#     # PANEL B: SEPARABILITY (Geometric Margin)
#     # ==========================================
#     ax1 = axes[1]
    
#     # Tạo dữ liệu: Khoảng cách giữa 2 tâm RẤT LỚN
#     mean_a, mean_b = 2.0, 8.0
#     d0_sep = np.random.normal(mean_a, 0.6, 50)
#     d1_sep = np.random.normal(mean_b, 0.6, 50)
    
#     # Vẽ Scatter Points (Jittered y) để thấy rõ vị trí
#     ax1.scatter(d0_sep, np.random.normal(1, 0.05, 50), color=c_cls0, alpha=0.6, s=30)
#     ax1.scatter(d1_sep, np.random.normal(1, 0.05, 50), color=c_cls1, alpha=0.6, s=30)
    
#     # Vẽ đường Mean (Centroids)
#     ax1.axvline(mean_a, color=c_cls0, linestyle='--', alpha=0.8, ymax=0.8)
#     ax1.axvline(mean_b, color=c_cls1, linestyle='--', alpha=0.8, ymax=0.8)
#     ax1.text(mean_a, 1.3, r'$\mu_A$', color=c_cls0, ha='center', fontweight='bold')
#     ax1.text(mean_b, 1.3, r'$\mu_B$', color=c_cls1, ha='center', fontweight='bold')
    
#     # Vẽ mũi tên Margin (Gap)
#     ax1.annotate('', xy=(mean_a, 1.15), xytext=(mean_b, 1.15),
#                  arrowprops=dict(arrowstyle='<->', color='black', lw=2))
#     ax1.text((mean_a+mean_b)/2, 1.2, 'Geometric Margin (Gap)', ha='center', va='bottom', fontweight='bold')
    
#     ax1.set_ylim(0.5, 1.5)
#     ax1.set_title('(B) Separability\n(Margin Maximization)', fontweight='bold')
#     ax1.set_xlabel('Distance Value')
#     ax1.set_yticks([])
    
#     # ==========================================
#     # PANEL C: INFORMATION GAIN (Split Purity)
#     # ==========================================
#     ax2 = axes[2]
    
#     # Tạo dữ liệu
#     d0_ig = np.random.normal(3, 0.5, 40) # Class A bên trái
#     d1_ig = np.random.normal(6, 0.5, 40) # Class B bên phải
#     threshold = 4.5
    
#     # Vẽ Points
#     ax2.scatter(d0_ig, np.random.normal(1, 0.05, 40), color=c_cls0, alpha=0.6, s=30)
#     ax2.scatter(d1_ig, np.random.normal(1, 0.05, 40), color=c_cls1, alpha=0.6, s=30)
    
#     # Vẽ Threshold Line (Decision Stump)
#     ax2.axvline(threshold, color=c_split, linestyle='-', linewidth=2.5)
#     ax2.text(threshold, 1.4, 'Split Threshold', color=c_split, ha='center', fontweight='bold')
    
#     # Vẽ vùng Purity (Entropy)
#     # Vùng trái
#     rect_left = patches.Rectangle((0, 0), threshold, 2, color=c_cls0, alpha=0.1)
#     ax2.add_patch(rect_left)
#     ax2.text(threshold - 1.5, 0.7, "Pure Node A\n(Entropy $\\approx$ 0)", ha='center', color=c_cls0, fontweight='bold')
    
#     # Vùng phải
#     rect_right = patches.Rectangle((threshold, 0), 10, 2, color=c_cls1, alpha=0.1)
#     ax2.add_patch(rect_right)
#     ax2.text(threshold + 1.5, 0.7, "Pure Node B\n(Entropy $\\approx$ 0)", ha='center', color=c_cls1, fontweight='bold')
    
#     ax2.set_xlim(1, 8)
#     ax2.set_ylim(0.5, 1.5)
#     ax2.set_title('(C) Information Gain\n(Split Purity)', fontweight='bold')
#     ax2.set_xlabel('Distance Value')
#     ax2.set_yticks([])

#     # Lưu ảnh
#     plt.tight_layout()
#     # plt.savefig('composite_score_components.png', dpi=300, bbox_inches='tight')
#     # print("Figure saved to composite_score_components.png")
#     plt.show()

# if __name__ == "__main__":
#     draw_composite_components()



import matplotlib.pyplot as plt
import numpy as np

def create_shapelet_grid():
    # Cấu hình style
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    
    # Tạo dữ liệu giả lập cho 4 Classes của BasicMotions (để demo)
    # Trong thực tế, bạn thay phần này bằng dữ liệu shapelet thật của bạn
    # BasicMotions có 6 chiều (3 Accel, 3 Gyro). Ở đây vẽ đại diện 1 chiều quan trọng nhất.
    x = np.linspace(0, 50, 50)
    
    # Class 0: Badminton (Cú đập mạnh - Spike)
    s_badminton = np.sin(x/5) * np.exp(-x/20) + 2.0 * np.exp(-((x-25)**2)/10)
    
    # Class 1: Running (Dao động biên độ lớn, tần số cao)
    s_running = 1.5 * np.sin(x/2)
    
    # Class 2: Standing (Dao động rất nhỏ, gần như phẳng)
    s_standing = 0.1 * np.random.randn(50)
    
    # Class 3: Walking (Dao động đều, biên độ trung bình)
    s_walking = 0.8 * np.sin(x/4)

    shapelets = [s_badminton, s_running, s_standing, s_walking]
    labels = ['(A) Badminton (Smash)', '(B) Running', '(C) Standing', '(D) Walking']
    colors = ['#d62728', '#1f77b4', '#7f7f7f', '#2ca02c'] # Đỏ, Xanh, Xám, Lục

    # Tạo Figure 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # Vẽ Shapelet
        ax.plot(shapelets[i], color=colors[i], linewidth=2.5)
        
        # Trang trí
        ax.set_title(labels[i], fontweight='bold', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Thêm vùng nền thể hiện "Discriminative Region"
        ax.axvspan(15, 35, color=colors[i], alpha=0.1)
        
        if i >= 2:
            ax.set_xlabel('Time Steps')
        if i % 2 == 0:
            ax.set_ylabel('Amplitude (Norm.)')

    plt.suptitle('Top-1 Discriminative Shapelets for BasicMotions Dataset', fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Lưu ảnh
    save_path = 'top_shapelets_basicmotions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined image to {save_path}")
    plt.show()

if __name__ == "__main__":
    create_shapelet_grid()