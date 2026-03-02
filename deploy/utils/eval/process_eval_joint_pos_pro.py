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

            # --- 核心计算 ---
            
            # 1. 计算所有数值列的均值
            all_means = df.mean(numeric_only=True)

            # 2. 计算 error_joint_pos 的方差
            ejp_var = df['error_joint_pos'].var()

            # 3. 计算 成功情况下的 error_joint_pos 均值
            if 'success' in df.columns and (df['success'] == 1.0).any():
                ejp_success_mean = df[df['success'] == 1.0]['error_joint_pos'].mean()
            else:
                # 如果没有成功样本或没有success列，设为0或NaN
                ejp_success_mean = 0.0

            # 4. 处理成功率：转为百分数数值
            if 'success' in df.columns:
                sr_raw = all_means.get('success', 0.0)
            else:
                sr_raw = (df['step'] >= 498).mean() # 兜底逻辑
            
            success_rate_value = round(float(sr_raw) * 100, 2)

            # --- 构造结果 ---

            # 打印列表顺序: [5个Error均值, Step均值, SuccessRate数值, EJP方差, EJP成功均值]
            print_results = []
            for col in error_cols:
                print_results.append(round(all_means.get(col, 0.0), 4))
            
            print_results.append(round(all_means.get('step', 0.0), 4))
            print_results.append(success_rate_value)
            print_results.append(round(float(ejp_var), 6)) # 方差通常较小，保留6位
            print_results.append(round(float(ejp_success_mean), 4))

            # --- 写入文件 ---
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write('\n--- Summary Statistics ---\n')
                summary_parts = [f"{col}_mean: {all_means.get(col, 0.0):.4f}" for col in error_cols]
                summary_parts.append(f"mean_step: {all_means.get('step', 0.0):.4f}")
                summary_parts.append(f"success_rate: {success_rate_value}")
                summary_parts.append(f"error_joint_pos_var: {ejp_var:.6f}")
                summary_parts.append(f"error_joint_pos_success_mean: {ejp_success_mean:.4f}")
                
                f.write(", ".join(summary_parts) + "\n")

            # --- 打印结果 ---
            print(f"File: {filename}")
            print(f"Metrics (Errors[5], Step, SR, EJP_Var, EJP_Succ_Mean):")
            print(print_results)
            print("-" * 30)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # 确保路径正确
    target_directory = "./logs/test"
    process_csv_directory(target_directory)