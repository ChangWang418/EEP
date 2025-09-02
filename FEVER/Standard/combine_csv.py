import os
import pandas as pd

# 设置你保存39个csv文件的文件夹路径
input_folder = r"C:\Users\sakur\Desktop\paper\fever-standard"  # ← 替换为你的路径
output_file = r"C:\Users\sakur\Desktop\paper\fever-standard\fever_dev_combined.csv" # ← 设置输出路径

# 遍历文件夹中所有csv文件，读取并存入列表
all_dfs = []
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)
        all_dfs.append(df)

# 合并所有DataFrame
combined_df = pd.concat(all_dfs, ignore_index=True)

# 保存为一个新的csv文件
combined_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"✅ 合并完成！共计 {len(combined_df)} 条记录，已保存到：\n{output_file}")
