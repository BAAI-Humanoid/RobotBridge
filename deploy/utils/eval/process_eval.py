import os
import glob
import pandas as pd

def process_csv_directory(directory_path):
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return

    # 指定误差项列名
    error_cols = [
        'error_anchor_pos', 
        # 'error_anchor_lin_vel', 
        # 'error_body_pos',
        'error_body_pos_w',
        # 'error_eef_pos',
        'error_eef_pos_w',
        # 'error_body_lin_vel', 
    ]

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)

            # 数据清洗与数值化
            df['step'] = pd.to_numeric(df['step'], errors='coerce')
            if 'success' in df.columns:
                df['success'] = pd.to_numeric(df['success'], errors='coerce')

            df = df.dropna(subset=['step'])
            if df.empty:
                continue

            # 1. 计算所有数值列的均值
            all_means = df.mean(numeric_only=True)

            # 2. 处理成功率：转为百分数数值 (例如 0.9989 -> 99.89)
            if 'success' in df.columns:
                sr_raw = all_means.get('success', 0.0)
            else:
                sr_raw = (df['step'] >= 498).mean()
            
            # 仅保留数值，不加 % 符号
            success_rate_value = round(float(sr_raw) * 100, 2)

            # 3. 构造打印列表
            # 顺序: [5个Error均值, Step均值, SuccessRate数值]
            print_results = []
            
            # 添加误差均值
            for col in error_cols:
                print_results.append(round(all_means.get(col, 0.0), 4))
            
            # 添加 Step 均值 (放在成功率前一列)
            print_results.append(round(all_means.get('step', 0.0), 4))
            
            # 添加 成功率数值
            print_results.append(success_rate_value)

            # # 4. 写入文件 (只写均值，不写最大值)
            # with open(file_path, 'a', encoding='utf-8') as f:
            #     f.write('\n--- Summary Statistics ---\n')
            #     # 按照相同顺序构建写入字符串
            #     summary_parts = [f"{col}_mean: {all_means.get(col, 0.0):.4f}" for col in error_cols]
            #     summary_parts.append(f"mean_step: {all_means.get('step', 0.0):.4f}")
            #     summary_parts.append(f"success_rate: {success_rate_value}")
                
            #     f.write(", ".join(summary_parts) + "\n")

            # 5. 打印结果
            print(f"File: {filename}")
            print(f"Metrics (Errors..., Mean_Step, Success_Rate_Num):")
            print(print_results)
            print("-" * 30)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # target_directory = "/home/ws/hbs/RobotBridge/deploy/logs/motionx_300_adaptation/noitom"
    target_directory = "/home/ws/hbs/RobotBridge/deploy/logs/noitom_test_dynamic"
    process_csv_directory(target_directory)
