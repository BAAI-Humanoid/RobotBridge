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

            # --- 新增：记录并打印成功列表 ---
            # 尝试找到一个可以作为“名称”或“ID”的列
            id_column = None
            for col in ['id', 'seed', 'episode', 'name', 'run']:
                if col in df.columns:
                    id_column = col
                    break
            
            # 获取成功的行掩码
            success_mask = df['step'] >= target_step
            
            # 根据标识列提取成功列表，如果没有标识列，则使用行号(从0开始)
            if id_column:
                success_list = df.loc[success_mask, id_column].tolist()
                id_desc = id_column
            else:
                success_list = df.index[success_mask].tolist()
                id_desc = "row_index"

            # 打印成功列表
            print(f"\n>>> File: {filename}")
            print(f"    Successful Records ({id_desc}): {success_list}")
            # -------------------------------

            # 2. 计算各项指标的均值 (排除早死记录的 error)
            error_cols = ['error_joint_pos', 'error_joint_vel']
            existing_error_cols = [col for col in error_cols if col in df.columns]
            
            df_for_mean = df.copy()
            if existing_error_cols:
                # 对于 step < 498 的记录，将其 error 设为 NaN，不参与 mean() 计算
                df_for_mean.loc[~success_mask, existing_error_cols] = np.nan
            
            means = df_for_mean.mean(numeric_only=True)
            
            # 3. 计算成功率
            total_rows = len(df)
            success_rows = success_mask.sum()
            success_rate = success_rows / total_rows if total_rows > 0 else 0
            
            # 4. 准备追加的内容
            mean_df = pd.DataFrame([means])
            
            # 5. 写入文件
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write('\n')
                # 追加均值行 (不带表头)
                mean_df.to_csv(f, header=False, index=False, lineterminator='\n')
                # 追加成功率统计
                f.write(f"Success Rate (step >= {target_step}), {success_rate:.2%}, ({success_rows}/{total_rows})\n")
                
            print(f"    Successfully processed. Success Rate: {success_rate:.2%}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # 修改为你的 CSV 目录路径
    target_directory = "./logs/joint_only_success" 
    process_csv_directory(target_directory)