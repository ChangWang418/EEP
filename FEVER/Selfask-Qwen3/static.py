import os
import pandas as pd
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "fever_dev_combined.csv")

df = pd.read_csv(file_path)
df.columns = [c.lower().strip() for c in df.columns]

# 统一大写
df["true_label"] = df["true_label"].str.upper()
df["predicted_label"] = df["predicted_label"].str.upper()

# 🚫 排除无效标签
INVALID_LABELS = {"ERROR", "NO_LABEL"}

df_valid = df[
    ~df["true_label"].isin(INVALID_LABELS) &
    ~df["predicted_label"].isin(INVALID_LABELS)
]

y_true = df_valid["true_label"]
y_pred = df_valid["predicted_label"]

print("📊 Classification Report (ERROR / NO_LABEL excluded):\n")
print(classification_report(y_true, y_pred, digits=3))
