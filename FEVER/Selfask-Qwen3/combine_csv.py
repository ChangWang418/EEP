import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_folder = BASE_DIR
output_file = os.path.join(BASE_DIR, "fever_dev_combined.csv")

def read_csv_robust(path):
    # 1) 常见：utf-8 / utf-8-sig
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        pass

    # 2) 中文 Windows 常见：gbk
    try:
        return pd.read_csv(path, encoding="gbk")
    except UnicodeDecodeError:
        pass

    # 3) 最后兜底：latin1 永远不报解码错（但可能会有乱码字符）
    return pd.read_csv(path, encoding="latin1")

all_dfs = []
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        df = read_csv_robust(file_path)
        all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)
combined_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"✅ Merge completed! Total records: {len(combined_df)}, saved to:\n{output_file}")
