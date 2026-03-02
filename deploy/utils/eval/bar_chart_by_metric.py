import matplotlib.pyplot as plt
import numpy as np

# --- 1. 数据准备 ---

# 【Group 1: 基础模型对比】
# file_names = [
#     "DAgger (W)", "Pure RL (W)", 
#     "Pure RL (R)", "TWIST", "GMT"
# ]
# raw_data = [
#     [716.6983, 75.83], [740.9305, 77.88], 
#     [706.7662, 73.46], [723.5577, 76.62], [615.1769, 61.45]
# ]

# 【Group 2: 数据源数量对比】
file_names = [
    "1-Source", "3-Sources", "5-Sources"
]
raw_data = [
    [279.8041, 14.69], [581.0979, 58.61], [654.8531, 67.77]
]

# 【Group 3: Noitom 适配对比】
# file_names = [
#     "Adapter Noitom 30(G)", "Adapter Noitom 30(L)", 
#     "Finetune Noitom 30(G)", "Continual Noitom 30(G)"
# ]
# raw_data = [
#     [740.5403, 76.62], [739.6888, 77.73], [0, 0], [740.3555, 78.67]
# ]

# # 【Group 4: Pico 方案对比】（当前激活）
# file_names = [
#     "Adapter Pico 40(G)", "Adapter Pico 40(L)", 
#     "Finetune Pico 30(G)", "Continual Pico 40(G)", "FLD Pico 30(G)"
# ]
# raw_data = [
#     [740.3791, 77.25], [0, 0], [0, 0], [745.9984, 78.36], [0, 0]
# ]

# 【Group 5: Pico 适配时长对比】
# file_names = [
#     "Adapter Pico 2min", "Adapter Pico 30min"
# ]
# raw_data = [
#     [737.9937, 77.25], [737.0885, 77.25]
# ]

# --- 数据解析 ---
ms_data = [d[0] for d in raw_data]  # Mean Step
sr_data = [d[1] for d in raw_data]  # Success Rate

# 颜色配置 (根据 file_names 长度自适应)
academic_colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452'][:len(file_names)]

# --- 2. 紧凑布局参数 ---
n_methods = len(file_names)
width = 0.035  
cluster_centers = [0.3, 0.55] 

fig, ax1 = plt.subplots(figsize=(10, 6), dpi=100)
ax2 = ax1.twinx()

# --- 3. 绘制柱状图 ---
for i in range(n_methods):
    offset = (i - (n_methods - 1) / 2) * width
    
    # Success Rate (左轴 - 百分比)
    b1 = ax1.bar(cluster_centers[0] + offset, sr_data[i], width, 
                 color=academic_colors[i], edgecolor='black', linewidth=0.4, alpha=0.9)
    
    # Mean Step (右轴 - 步数)
    b2 = ax2.bar(cluster_centers[1] + offset, ms_data[i], width, 
                 color=academic_colors[i], edgecolor='black', linewidth=0.4, alpha=0.9)

    # 柱顶数值
    ax1.bar_label(b1, padding=3, fmt='%.1f', fontsize=14, fontweight='bold')
    ax2.bar_label(b2, padding=3, fmt='%.0f', fontsize=14, fontweight='bold')

# --- 4. Y 轴范围逻辑 ---
# sr_max = max(sr_data) if max(sr_data) > 0 else 100
# ax1.set_ylim(sr_max / 2, sr_max * 1.3) 

# ms_max = max(ms_data) if max(ms_data) > 0 else 1000
# ax2.set_ylim(ms_max / 2, ms_max * 1.3)
sr_max = max(sr_data) if max(sr_data) > 0 else 100
ax1.set_ylim(0, sr_max * 1.5) 

ms_max = max(ms_data) if max(ms_data) > 0 else 1000
ax2.set_ylim(0, ms_max * 1.5)

# --- 5. 坐标轴与字体微调 ---
ax1.set_xticks(cluster_centers)
ax1.set_xticklabels(['Success Rate (%)', 'Mean Step'], fontsize=12, fontweight='bold')
ax1.set_xlim(cluster_centers[0] - 0.15, cluster_centers[1] + 0.15)

ax1.set_ylabel('Success Rate (%)', fontsize=14)
ax2.set_ylabel('Mean Step', fontsize=14)
ax1.tick_params(axis='both', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)

ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.yaxis.grid(True, linestyle='--', alpha=0.2)

# --- 6. 图例设置 ---
ax1.legend(handles=[plt.Rectangle((0,0),1,1, color=academic_colors[i]) for i in range(n_methods)], 
           labels=file_names,
           title="Methods",
           loc='upper right', 
           bbox_to_anchor=(0.96, 1.2), # 向左微移 (1.0 是最右边)
           fontsize=14,               # 【修改2：增大图例字体】
           title_fontsize=16,          # 【修改3：增大图例标题】
           frameon=True, 
           framealpha=0.8,
           edgecolor='lightgrey')

# plt.title('Evaluation Analysis', fontsize=12, fontweight='bold', pad=15)

fig.tight_layout()
plt.show()