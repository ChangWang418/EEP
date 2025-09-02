import pandas as pd
from sklearn.metrics import classification_report

# 1. 加载合并后的CSV文件
file_path = r"C:\Users\sakur\Desktop\paper\Pubhealt4label-decompose\pubhealth_dev_combined.csv"  # 改成你的路径
df = pd.read_csv(file_path)

# 2. 统一列名小写处理
df.columns = [c.lower().strip() for c in df.columns]

# 3. 提取真实标签和预测标签，并统一为大写
y_true = df["true_label"].str.upper()
y_pred = df["predicted_label"].str.upper()

# 4. 过滤掉含有 ERROR 的样本
mask = (y_true != "ERROR") & (y_pred != "ERROR")
y_true_filtered = y_true[mask]
y_pred_filtered = y_pred[mask]

# 5. 打印分类报告
print("📊 Classification Report (excluding ERROR):\n")
print(classification_report(y_true_filtered, y_pred_filtered, digits=3))

