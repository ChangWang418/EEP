import os
import json
import sqlite3
import csv  # ✅ Added
from model.gpt_verifier import gpt_self_ask_verifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from dotenv import load_dotenv
import os
import openai

load_dotenv()  # By default, read the .env file from the project root directory
openai.api_key = os.getenv("OPENAI_API_KEY")

# Directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Point to the Fever_Data folder
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Fever_Data"))

# Paths to database and JSON files
FEVER_DB_PATH = os.path.join(DATA_DIR, "fever.db")
fever_jsonl = os.path.join(DATA_DIR, "shared_task_dev.jsonl")

# Output directory (can be the same as DATA_DIR or a sibling directory)
OUTPUT_DIR = BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create if it doesn't exist

# Get the sentence for a given page_title and sentence index from fever.db
def get_sentence_from_db(page_title, sentence_index, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT lines FROM documents WHERE id = ?", (page_title,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return f"[Page not found: {page_title}]"

    lines = row[0].split('\n')
    for l in lines:
        if '\t' in l:
            idx, sentence = l.split('\t', 1)
            try:
                if int(idx) == sentence_index:
                    return sentence
            except ValueError:
                continue
    return f"[Sentence index {sentence_index} not found in {page_title}]"

# Load training or validation data and extract sentences based on evidence
def load_fever_data_with_db(jsonl_path, db_path, limit=None):
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            item = json.loads(line)
            claim = item['claim']
            label = item.get('label', 'NOT ENOUGH INFO').upper()
            evidences = item.get('evidence', [])
            if evidences:
                first_group = evidences[0]
                sentences = []
                for e in first_group:
                    if len(e) >= 4 and isinstance(e[3], int):
                        page_title = e[2]
                        sentence_idx = e[3]
                        sentence = get_sentence_from_db(page_title, sentence_idx, db_path)
                        sentences.append(sentence)
                evidence_text = " ".join(sentences)
            else:
                evidence_text = "[No evidence provided]"
            data.append((claim, evidence_text, label))
    return data

def run():
    import os
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    import matplotlib.pyplot as plt

    data = load_fever_data_with_db(fever_jsonl, FEVER_DB_PATH, limit=None)

    y_true, y_pred = [], []
    batch_size = 300

    # ✅ Check how many CSV files already exist (in the specified directory)
    file_index = 0
    while os.path.exists(os.path.join(OUTPUT_DIR, f"fever_dev_result_{file_index}.csv")):
        file_index += 1

    start_index = file_index * batch_size
    print(f"🔁 Resuming from file {file_index} / item index {start_index}")
    data = data[start_index:]

    writer, f = None, None

    for i, (claim, evidence, label) in enumerate(data):
        global_index = start_index + i

        # Open a new file every batch_size items
        if i % batch_size == 0:
            if f:
                f.close()
            csv_path = os.path.join(OUTPUT_DIR, f"fever_dev_result_{file_index}.csv")
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
