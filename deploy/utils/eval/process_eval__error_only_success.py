import pandas as pd
import os
import glob
import numpy as np

def process_csv_directory(directory_path, target_step=498):
    # 获取目录下所有的 csv 文件
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        try:
            # 1. 读取数据
            df = pd.read_csv(file_path)
            
            # 过滤掉非数据行（比如之前追加的统计行）
            df['step'] = pd.to_numeric(df['step'], errors='coerce')
            df = df.dropna(subset=['step'])
            
            if df.empty:
                print(f"Skipping empty file: {filename}")
                continue

            # 获取成功的行掩码 (活着)
            success_mask = df['step'] >= target_step
            total_rows = len(df)
            success_rows = success_mask.sum()
            success_rate = success_rows / total_rows if total_rows > 0 else 0

            # 2. 计算各项指标的均值
            # 创建一个用于计算均值的临时副本
            df_for_mean = df.copy()
            
            # 找出所有的 error 相关列（除了 step 以外的所有数值列）
            all_cols = df.columns.tolist()
            # 排除 step 列，剩下的就是我们需要“只计算活着”的列
            metric_cols = [col for col in all_cols if col != 'step' and pd.api.types.is_numeric_dtype(df[col])]
            
            # 【关键修改】：除了 step 以外，其他 error 列在不符合 success_mask 时设为 NaN
            # 这样 mean() 函数会自动忽略这些掉进坑里的“死掉”的数据
            if metric_cols:
                df_for_mean.loc[~success_mask, metric_cols] = np.nan
            
            # 计算均值：
            # step 是基于 df (全样本) 计算的
            # metric_cols 是基于 df_for_mean (已屏蔽死掉的样本) 计算的
            means = df_for_mean.mean(numeric_only=True)
            
            # 3. 准备追加的内容
            # means 已经包含了 step(all) 和 metrics(alive only)
            mean_df = pd.DataFrame([means])
            
            # 打印调试信息
            print(f"\n>>> File: {filename}")
            print(f"    Total: {total_rows}, Success: {success_rows}, Rate: {success_rate:.2%}")
            if success_rows > 0:
                print(f"    Joint Pos Error (Alive only): {means.get('error_joint_pos', 'N/A')}")
            else:
                print(f"    Warning: No rows reached {target_step}, error metrics will be NaN.")

            # 4. 写入文件
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write('\n')
                # 追加均值行 (不带表头)
                mean_df.to_csv(f, header=False, index=False, lineterminator='\n')
                # 追加成功率统计
                f.write(f"Success Rate (step >= {target_step}), {success_rate:.2%}, ({success_rows}/{total_rows})\n")
                
            print(f"    Successfully processed.")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # 修改为你的实际 CSV 目录路径
    target_directory = "./logs/original" 
    process_csv_directory(target_directory)