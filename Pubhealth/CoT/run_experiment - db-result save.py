import os
import csv
from model.gpt_verifier import gpt_self_ask_verifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import openai

load_dotenv()  # By default, load the .env file from the project root directory
openai.api_key = os.getenv("OPENAI_API_KEY")

# Directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the dataset (Pubhealth_Data in the parent directory)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Pubhealth_Data"))
pubhealth_tsv = os.path.join(DATA_DIR, "dev.tsv")

# Output directory (set as a sibling "results" folder)
OUTPUT_DIR = BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Automatically create output directory

# Load PubHealth tsv data
def load_pubhealth_data(tsv_path, limit=None):
    data = []
    valid_labels = {"TRUE", "FALSE", "UNPROVEN", "MIXTURE"}  # ✅ Add list of valid labels
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            if not row.get("claim") or not row.get("main_text") or not row.get("label"):
                print(f"⚠️ Skipping row {i} due to missing fields.")
                continue
            label = row["label"].upper()
            if label not in valid_labels:
                print(f"⚠️ Skipping row {i} due to invalid label: {label}")
                continue  # ✅ Skip invalid label

            claim = row["claim"]
            evidence = row["main_text"]
            data.append((claim, evidence, label))
    return data

def run():
    data = load_pubhealth_data(pubhealth_tsv, limit=None)

    y_true, y_pred = [], []
    batch_size = 100

    # ✅ Check number of existing CSV files (in the specified directory)
    file_index = 0
    while os.path.exists(os.path.join(OUTPUT_DIR, f"pubhealth_dev_result_{file_index}.csv")):
        file_index += 1

    start_index = file_index * batch_size
    print(f"🔁 Resuming from file {file_index} / item index {start_index}")
    data = data[start_index:]

    writer, f = None, None

    for i, (claim, evidence, label) in enumerate(data):
        global_index = start_index + i

        # Create a new file every batch_size items
        if i % batch_size == 0:
            if f:
                f.close()
            csv_path = os.path.join(OUTPUT_DIR, f"pubhealth_dev_result_{file_index}.csv")
            f = open(csv_path, "w", encoding="utf-8", newline="")
            writer = csv.writer(f)
            writer.writerow(["Claim", "Evidence", "True_Label", "Predicted_Label"])
            print(f"\n📁 Writing to {csv_path}")
            file_index += 1

        try:
            prediction = gpt_self_ask_verifier(claim, evidence).upper()
        except Exception as e:
            print(f"❌ Error at item {global_index}: {e}")
            prediction = "ERROR"

        y_true.append(label)
        y_pred.append(prediction)
        writer.writerow([claim, evidence, label, prediction])

        print(f"{global_index + 1}")
        print(f"Claim: {claim}\nLabel: {label}\nPredicted: {prediction}\n{'-' * 60}")

    if f:
        f.close()

    # ✅ Output evaluation report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, digits=3)
    print(report)

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f_report:
        f_report.write(report)

    # ✅ Plot F1 score chart
    labels = sorted(set(y_true + y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    x = range(len(labels))
    plt.figure(figsize=(10, 5))
    plt.bar(x, f1, width=0.4, label='F1 Score')
    plt.xticks(x, labels)
    plt.ylabel("F1 Score")
    plt.title("F1 Score per Label")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "f1_score_chart.png"))
    plt.show()

    print("✅ Done. All resumed, saved and visualized.")

if __name__ == "__main__":
    run()
