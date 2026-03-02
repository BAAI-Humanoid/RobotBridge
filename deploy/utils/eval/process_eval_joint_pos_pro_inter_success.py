import os
import glob
import pandas as pd
import numpy as np

def process_csv_directory(directory_path):
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return

    # 指定误差项列名
    error_cols = [
        'error_anchor_pos', 
        'error_anchor_lin_vel', 
        'error_body_lin_vel', 
        'error_joint_vel', 
        'error_joint_pos'
    ]

    # --- 第一阶段：寻找所有文件成功轨迹的交集索引 ---
    common_success_indices = None
    
    print("Finding common success intersection...")
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            # 获取当前文件成功的行索引
            current_success_indices = set(df[df['success'] == 1.0].index)
            
            if common_success_indices is None:
                common_success_indices = current_success_indices
            else:
                common_success_indices = common_success_indices.intersection(current_success_indices)
        except Exception as e:
            print(f"Error reading {file_path} for intersection: {e}")

    if common_success_indices:
        common_list = sorted(list(common_success_indices))
        print(f"Found {len(common_list)} common successful trajectories.")
    else:
        common_list = []
        print("Warning: No common successful trajectories found across all files.")

    # --- 第二阶段：正式处理每个文件 ---
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)

            # 数据清洗与数值化
            df['step'] = pd.to_numeric(df['step'], errors='coerce')
            for col in error_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            if 'success' in df.columns:
                df['success'] = pd.to_numeric(df['success'], errors='coerce')

            df = df.dropna(subset=['step'])
            if df.empty:
                continue

            # 1. 常规全样本均值
            all_means = df.mean(numeric_only=True)

            # 2. 成功情况下的误差均值 (该文件自己成功的)
            ejp_success_mean = df[df['success'] == 1.0]['error_joint_pos'].mean() if (df['success'] == 1.0).any() else 0.0

            # 3. 核心功能：交集样本上的误差均值
            if common_list:
                # 确保索引不越界（针对某些文件行数不一致的防御性处理）
                valid_indices = [i for i in common_list if i in df.index]
                ejp_intersection_mean = df.loc[valid_indices, 'error_joint_pos'].mean()
            else:
                ejp_intersection_mean = 0.0

            # 4. 计算方差和成功率
            ejp_var = df['error_joint_pos'].var()
            sr_raw = all_means.get('success', 0.0)
            success_rate_value = round(float(sr_raw) * 100, 2)

            # --- 构造结果列表 ---
            # 顺序: [5个Error, Step, SR, EJP方差, EJP成功均值, EJP交集均值]
            print_results = [
                *[round(all_means.get(col, 0.0), 4) for col in error_cols],
                round(all_means.get('step', 0.0), 4),
                success_rate_value,
                round(float(ejp_var), 6),
                round(float(ejp_success_mean), 4),
                round(float(ejp_intersection_mean), 4)
            ]

            # --- 写入文件 ---
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write('\n--- Summary Statistics ---\n')
                summary_parts = [f"{col}_mean: {all_means.get(col, 0.0):.4f}" for col in error_cols]
                summary_parts.append(f"mean_step: {all_means.get('step', 0.0):.4f}")
                summary_parts.append(f"success_rate: {success_rate_value}")
                summary_parts.append(f"error_joint_pos_var: {ejp_var:.6f}")
                summary_parts.append(f"error_joint_pos_success_mean: {ejp_success_mean:.4f}")
                summary_parts.append(f"error_joint_pos_intersection_mean: {ejp_intersection_mean:.4f}")
                
                f.write(", ".join(summary_parts) + "\n")

            # --- 打印结果 ---
            print(f"File: {filename}")
            print(f"Metrics (Errors[5], Step, SR, Var, Succ_M, Inter_M):")
            print(print_results)
            print("-" * 30)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    target_directory = "./logs/test"
    process_csv_directory(target_directory)