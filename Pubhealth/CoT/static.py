import os
import pandas as pd
from sklearn.metrics import classification_report

# 1. Get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Load the merged CSV file from the current directory
file_path = os.path.join(BASE_DIR, "pubhealth_dev_combined.csv")
df = pd.read_csv(file_path)

# 3. Normalize column names to lowercase
df.columns = [c.lower().strip() for c in df.columns]

# 4. Extract true and predicted labels, and convert them to uppercase
y_true = df["true_label"].str.upper()
y_pred = df["predicted_label"].str.upper()

# 5. Print classification report
print("📊 Classification Report:\n")
print(classification_report(y_true, y_pred, digits=3))
