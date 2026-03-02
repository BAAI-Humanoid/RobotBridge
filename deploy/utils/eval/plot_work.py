import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

def generate_optimized_benchmark_report():
    # --- 1. Dataset Configuration ---
    metric_labels = [
        'Error Anchor Pos\n(m) ↓', 
        'Error Anchor Lin Vel\n(m/s) ↓', 
        # 'Error Body Pos Base\n(m) ↓', 
        'Error Body Pos World\n(m) ↓', 
        # 'Error EEF Pos Base\n(m) ↓', 
        'Error EEF Pos World\n(m) ↓', 
        'Error Body Lin Vel\n(m/s) ↓', 
        # 'Mean Step\n(length) ↑',
        # 'Success Rate\n(%) ↑',
    ]
    num_metrics = len(metric_labels)

    # ================= 1. 基础模型对比 (GMT, Twist, One-Stage, Distillation) =================
    # performance_results = {
    #     'GMT':                                     [9.5536, 3.0843, 0.5505, 9.5143, 0.7056, 9.4161, 3.0971, 633.64, 24.0],
    #     'Twist':                                   [3.919, 1.482, 0.2124, 3.9298, 0.2726, 3.947, 2.4469, 921.64, 100.0],
    #     'One-Stage 24K Global':                    [2.9352, 1.1682, 0.0497, 2.9353, 0.0808, 2.9363, 1.3946, 921.64, 100.0],
    #     'One-Stage 24K Local':                     [4.2667, 1.0769, 0.0444, 4.2651, 0.074, 4.2638, 1.2456, 921.64, 100.0],
    #     'Distillation Global (Student 22K)':       [2.5461, 1.1967, 0.0447, 2.5468, 0.0714, 2.5478, 1.4045, 921.64, 100.0],
    # }

    # ================= 2. 数据源数量对比 (1, 3, 5 Sources) =================
    # 注：提供的数据中仅包含 5 sources 相关数据
    # performance_results = {
    #     '1_source':                                [6.7584, 3.6103, 0.5756, 6.6798, 0.6204, 6.5132, 3.2163, 198.96, 0.0], # 数据缺失
    #     '3_sources':                               [5.0092, 1.3332, 0.1361, 5.0149, 0.1786, 5.0154, 1.5343, 855.0, 84.0], # 数据缺失
    #     '5_sources (One-Stage 24K)':               [2.9697, 1.1735, 0.0445, 2.9688, 0.0723, 2.9688, 1.3416, 921.64, 100.0],
    # }

    # ================= 3. Noitom 适配方案对比 (30min) =================
    # performance_results = {
    #     'Adapter Noitom 30min (G)':                 [3.1862, 1.1698, 0.0504, 3.1859, 0.0818, 3.1865, 1.3889, 921.64, 100.0],
    #     'Adapter Noitom 30min (L)':                 [2.9898, 1.1658, 0.0495, 2.9897, 0.0805, 2.9905, 1.3879, 921.64, 100.0],
    #     'Finetune Noitom 30min (G)':                [2.1272, 1.3837, 0.0672, 2.1289, 0.0992, 2.1367, 1.642, 901.28, 96.0],
    #     'Continual Noitom 30min (G)':               [3.0942, 1.177, 0.0509, 3.0936, 0.0828, 3.0945, 1.4031, 921.64, 100.0],
    # }

    # ================= 4. Pico 适配方案对比 (30/40min) =================
    # performance_results = {
    #     'Adapter Pico 40min (G)':                   [1.194, 1.3565, 0.0515, 1.1961, 0.0843, 1.2002, 1.6067, 921.64, 100.0],
    #     'Adapter Pico 40min (L)':                   [1.5278, 1.3791, 0.067, 1.537, 0.0991, 1.5436, 1.6384, 904.56, 96.0],
    #     'Finetune Pico 30min (G)':                  [1.4106, 1.5335, 0.0803, 1.4043, 0.1308, 1.3998, 1.8167, 885.68, 92.0],
    #     'Continual Pico 40min (G)':                 [1.7248, 1.3014, 0.0515, 1.7272, 0.082, 1.7294, 1.5526, 921.64, 100.0],
    #     'FLD Pico 30min (G)':                       [2.9298, 1.1536, 0.0494, 2.9298, 0.0802, 2.9307, 1.3827, 921.64, 100.0],
    # }
    # performance_results = {
    #     'Adapter Pico 30min (G)':                   [1.194, 1.3565, 1.1961, 1.2002, 1.6067, 921.64, 100.0],
    #     'Adapter Pico 30min (L)':                   [1.5278, 1.3791, 1.537, 1.5436, 1.6384, 904.56, 96.0],
    #     'Finetune Pico 30min (G)':                  [1.4106, 1.5335, 1.4043, 1.3998, 1.8167, 885.68, 92.0],
    #     'Continual Pico 30min (G)':                 [1.7248, 1.3014, 1.7272, 1.7294, 1.5526, 921.64, 100.0],
    #     'FLD Pico 30min (G)':                       [2.9298, 1.1536, 2.9298, 2.9307, 1.3827, 921.64, 100.0],
    # }

    # ================= 5. Pico 适配时长对比 (2min vs 30min) =================
    # performance_results = {
    #     'Adapter Pico 2min (G)':                    [2.8254, 1.1739, 0.0498, 2.8257, 0.0809, 2.8269, 1.3976, 921.64, 100.0],
    #     'Adapter Pico 30min (G)':                   [1.9982, 1.2852, 0.0633, 1.9991, 0.0983, 2.0002, 1.5619, 916.88, 96.0],
    # }


    # ================= 1. 基础模型对比 (GMT, Twist, One-Stage, Distillation) =================
    performance_results = {
        'GMT':                 [2.02, 1.27, 2.06, 2.02, 1.45],
        'Twist':               [1.34, 0.90, 1.39, 1.37, 2.52],
        'Pure RL (W)':         [0.82, 0.57, 0.85, 0.86, 0.79],
        'Pure RL (R)':         [1.20, 0.70, 1.22, 1.22, 0.89],
        'DAgger (W)':          [0.93, 0.62, 0.95, 0.95, 0.87],
    }

    # # ================= 2. 数据源数量对比 (1, 3, 5 Sources) =================
    # performance_results = {
    #     '1-Source':            [3.20, 2.43, 3.19, 3.07, 2.78],
    #     '3-Sources':           [1.60, 1.15, 1.62, 1.59, 1.39], 
    #     '5-Sources':           [1.34, 0.84, 1.36, 1.35, 1.11],
    # }

    # ================= 3. Noitom 适配方案对比 (30min) =================
    # performance_results = {
    #     'Adapter Noitom 30min (G)':   [0.85, 0.61, 0.13, 0.87, 0.17, 0.88, 0.87],
    #     'Adapter Noitom 30min (L)':   [0.83, 0.59, 0.13, 0.85, 0.18, 0.86, 0.82],
    #     'Fintune Noitom 30min (G)':   [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    #     'Continual Noitom 30min (G)': [0.76, 0.62, 0.14, 0.79, 0.19, 0.80, 0.85],
    # }

    # ================= 4. Pico 方案对比 (30/40min) =================
    # performance_results = {
    #     'Adapter Pico 40min (G)':     [0.82, 0.58, 0.13, 0.84, 0.17, 0.85, 0.80],
    #     'Adapter Pico 40min (L)':     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    #     'Fintune Pico 30min (G)':     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    #     'Continual Pico 40min (G)':   [0.86, 0.60, 0.12, 0.88, 0.17, 0.89, 0.82],
    #     'Fld Pico 30min (G)':         [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    # }

    # ================= 5. Pico 适配时长对比 (2min vs 30min) =================
    # performance_results = {
    #     'Adapter Pico 2min':          [0.83, 0.60, 0.13, 0.86, 0.18, 0.87, 0.81],
    #     'Adapter Pico 30min':         [0.83, 0.59, 0.12, 0.85, 0.17, 0.87, 0.80],
    # }


    # --- 2. Advanced Scaling Strategy ---
    raw_matrix = np.array(list(performance_results.values()))
    metric_mins = raw_matrix.min(axis=0)
    metric_maxs = raw_matrix.max(axis=0)

    num_col = raw_matrix.shape[-1]

    def get_visual_scores(values):
        scores = []
        for i, val in enumerate(values):
            # if i < num_col-2: # Error metrics: lower is better
            #     gap = (val - metric_mins[i]) / (metric_maxs[i] - metric_mins[i] + 1e-6)
            # else: # Performance metrics: higher is better
            #     gap = (metric_maxs[i] - val) / (metric_maxs[i] - metric_mins[i] + 1e-6)
            gap = (val - metric_mins[i]) / (metric_maxs[i] - metric_mins[i] + 1e-6)
            
            # Non-linear scaling (0.8) to distinguish high-end performance
            score = 1.0 - (gap ** 0.8) * 0.6
            scores.append(score)
        return scores

    model_names = list(performance_results.keys())
    visual_scores = {name: get_visual_scores(performance_results[name]) for name in model_names}

    # --- 3. Geometric Setup ---
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles_poly = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(12, 13), subplot_kw=dict(polar=True))
    
    color_palette = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', 
                     '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF']

    # ax.set_ylim(0, 1.35) 
    ax.set_ylim(0, 1.1)

    for idx, name in enumerate(model_names):
        color = color_palette[idx % len(color_palette)]
        scores = visual_scores[name]
        path = scores + [scores[0]]
        ax.plot(angles_poly, path, color=color, linewidth=3.2, label=name, zorder=3)
        ax.fill(angles_poly, path, color=color, alpha=0.04)

    # --- 4. Vertex-Avoidance Staggered Labeling ---
    # for axis_idx in range(num_metrics):
    #     theta = angles[axis_idx]
    #     axis_pts = []
    #     for model_idx, name in enumerate(model_names):
    #         axis_pts.append({
    #             'score': visual_scores[name][axis_idx],
    #             'raw': performance_results[name][axis_idx],
    #             'color': color_palette[model_idx % len(color_palette)]
    #         })
        
    #     axis_pts.sort(key=lambda x: x['score'])
        
    #     # Collision logic parameters
    #     min_radial_gap = 0.05
    #     current_radial_pos = -10.0
        
    #     deg = np.rad2deg(theta) % 360
    #     # Alignment logic based on quadrant
    #     if 345 <= deg or deg <= 15:   ha, va = 'left', 'center'
    #     elif 15 < deg < 165:          ha, va = 'center', 'bottom'
    #     elif 165 <= deg <= 195:       ha, va = 'right', 'center'
    #     else:                         ha, va = 'center', 'top'

    # # --- Universal Zipper Logic: Move text away from the vertex point ---
    #     for rank, pt in enumerate(axis_pts):
    #         # 1. Radial Buffer: Start further out from the vertex (0.05 vs 0.025)
    #         ideal_pos = pt['score'] + 0.05 
    #         actual_radial_pos = max(ideal_pos, current_radial_pos + min_radial_gap)
    #         current_radial_pos = actual_radial_pos
            
    #         # 2. Tangential Nudge: Apply to ALL labels to "hug" the axis rather than cover it
    #         # This ensures the data point at the vertex remains visible.
    #         nudge_magnitude = 0.05 
    #         tangential_nudge = nudge_magnitude * (1 if rank % 2 == 0 else -1)
            
    #         label_text = f"{pt['raw']:.2f}"
    #         # if axis_idx == 5: label_text += "%"
            
    #         t = ax.text(theta + tangential_nudge, actual_radial_pos, label_text, color=pt['color'],
    #                     fontsize=12, fontweight='bold', ha=ha, va=va,
    #                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.1))
            
    #         t.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground='white', alpha=0.9)])

    # --- 5. Aesthetics, Titles, and Legend ---
    ax.set_xticks(angles)
    ax.set_xticklabels([])

    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], labels=[])
    ax.grid(True, axis='y', linestyle='--', color='#BBBBBB', alpha=0.5)
    ax.grid(True, axis='x', linestyle='-', color='#EEEEEE', alpha=0.4)

    # TITLES: Fixed at radius 1.32
    # for i, (angle, label) in enumerate(zip(angles, metric_labels)):
    #     ax.text(angle, 1.45, label, size=11, ha='center', va='center', 
    #             fontweight='bold', color='#111111')

    ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False) 

    # LEGEND: Increased font size to 15
    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.16), 
    #            ncol=3, frameon=False, fontsize=15, columnspacing=2.5)
    # plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), 
    #            ncol=1, frameon=True, fontsize=11)
    plt.legend(
        loc='upper left', 
        bbox_to_anchor=(0.85, 1.25),  # 第一个值越小越往左，第二个值越大越往上
        ncol=1, 
        frameon=False,               # 建议去掉边框，视觉上更清爽
        fontsize=24
    )

    # 2. 【关键步】调整画布边距，确保图例不会被切掉
    # right=0.8 表示图形主体只占据画布左侧80%的空间，给右侧留出20%放图例
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)


    # plt.title("Motion Tracking Comprehensive Benchmark\nMultiple Works Comparison", 
    #           size=14, y=1.08, fontweight='bold', pad=20)
    
    plt.subplots_adjust(bottom=0.18, top=0.82)
    plt.show()

if __name__ == "__main__":
    generate_optimized_benchmark_report()