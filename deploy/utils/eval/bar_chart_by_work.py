import matplotlib.pyplot as plt
import numpy as np

# --- 1. 准备数据 ---
works = ['Method A', 'Method B', 'Method C']
success_rates = [85, 92, 78]      # 单位 %
mean_steps = [12.5, 10.2, 15.6]   # 无单位

# x = np.arange(len(works))  # 标签位置
# width = 0.35               # 柱状图宽度

# fig, ax = plt.subplots(figsize=(8, 6))

# # --- 2. 绘制柱状图 ---
# # 这里的颜色是按“指标”区分的
# rects1 = ax.bar(x - width/2, success_rates, width, label='Success Rate (%)', color='#3498db')
# rects2 = ax.bar(x + width/2, mean_steps, width, label='Mean Step', color='#e74c3c')

# # --- 3. 设置标签和样式 ---
# ax.set_ylabel('Value')
# ax.set_title('Performance Comparison by Method')
# ax.set_xticks(x)
# ax.set_xticklabels(works)
# ax.legend()

# # 在柱子上添加数值标签
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

# fig.tight_layout()
# plt.show()

def plot_version_1():
    # 创建两个子图，共享 X 轴（可选）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    x = np.arange(len(works))
    width = 0.5

    # --- 绘制 Success Rate ---
    rects1 = ax1.bar(x, success_rates, width, color='#3498db', label='Success Rate')
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Performance Metrics by Method', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(works)
    ax1.set_ylim(0, 110) # 比例通常到 100，留点空间标数字
    ax1.bar_label(rects1, padding=3, fmt='%.1f%%') # 标注具体值

    # --- 绘制 Mean Step ---
    rects2 = ax2.bar(x, mean_steps, width, color='#e67e22', label='Mean Step')
    ax2.set_ylabel('Mean Step (count)', fontsize=12)
    ax2.set_xlabel('Methods', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(works)
    # 根据数据动态调整 Y 轴，留出 15% 空间给数字
    ax2.set_ylim(0, max(mean_steps) * 1.15) 
    ax2.bar_label(rects2, padding=3, fmt='%.1f') # 标注具体值

    fig.tight_layout()
    plt.show()

plot_version_1()