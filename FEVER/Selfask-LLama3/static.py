import os
import pandas as pd
from sklearn.metrics import classification_report

# 1. Get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Load the merged CSV file from the current directory
file_path = os.path.join(BASE_DIR, "fever_dev_combined.csv")
df = pd.read_csv(file_path)

# 3. Normalize column names to lowercase
df.columns = [c.lower().strip() for c in df.columns]

# 4. Filter out rows where 'true_label' or 'predicted_label' contains 'error'
df = df[~df["true_label"].str.contains("error", case=False, na=False)]
df = df[~df["predicted_label"].str.contains("error", case=False, na=False)]

# 5. Extract true and predicted labels, and convert them to uppercase
y_true = df["true_label"].str.upper()
y_pred = df["predicted_label"].str.upper()

# 6. Print classification report
print("📊 Classification Report:\n")
print(classification_report(y_true, y_pred, digits=3))
