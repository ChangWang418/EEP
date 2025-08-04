import os
import json
import sqlite3
import csv
import re
import random
from collections import defaultdict
from model.gpt_verifier import gpt_self_ask_verifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import openai

load_dotenv()  # By default, read the .env file from the project root directory
openai.api_key = os.getenv("OPENAI_API_KEY")

random.seed(42)

# Directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the Fever_Data folder
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Fever_Data"))

# Paths to database and JSON files
FEVER_DB_PATH = os.path.join(DATA_DIR, "fever.db")
fever_jsonl = os.path.join(DATA_DIR, "shared_task_dev.jsonl")

# Output directory (can be the same as DATA_DIR or a sibling directory)
OUTPUT_DIR = BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create if it doesn't exist

# Retrieve a sentence from the fever.db using page title and sentence index
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

# Load FEVER data and extract supporting evidence from database
def load_fever_data_with_db(jsonl_path, db_path, per_class_limit=None):
    data_by_label = defaultdict(list)
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
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
            data_by_label[label].append((claim, evidence_text, label))
    final_data = []
    for label, items in data_by_label.items():
        sampled = random.sample(items, min(len(items), per_class_limit))
        final_data.extend(sampled)
    random.shuffle(final_data)
    return final_data

# Extract label from model output
def extract_label_from_output(output_text):
    m = re.search(r"\b(SUPPORTS|REFUTES|NOT ENOUGH INFO)\b", output_text.upper())
    return m.group(1) if m else "NO_LABEL"

def run():
    data = load_fever_data_with_db(fever_jsonl, FEVER_DB_PATH, per_class_limit=666)

    y_true, y_pred = [], []
    batch_size = 300

    # Resume from last written file
    file_index = 0
    while os.path.exists(os.path.join(OUTPUT_DIR, f"fever_dev_result_{file_index}.csv")):
        file_index += 1

    start_index = file_index * batch_size
    print(f"🔁 Resuming from file {file_index} / item index {start_index}")
    data = data[start_index:]

    writer, f = None, None

    for i, (claim, evidence, label) in enumerate(data):
        global_index = start_index + i

        # Create new CSV file for each batch
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
            raw_output = gpt_self_ask_verifier(claim, evidence)
            prediction = extract_label_from_output(raw_output)
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

    # Filter only valid predicted labels
    valid_labels = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
    filtered_y_true = [yt for yt, yp in zip(y_true, y_pred) if yp in valid_labels]
    filtered_y_pred = [yp for yp in y_pred if yp in valid_labels]

    print("\nClassification Report (Filtered):")
    report = classification_report(filtered_y_true, filtered_y_pred, labels=valid_labels, digits=3)
    print(report)

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f_report:
        f_report.write(report)

    # Plot F1 scores per label
    precision, recall, f1, _ = precision_recall_fscore_support(filtered_y_true, filtered_y_pred, labels=valid_labels, zero_division=0)

    x = range(len(valid_labels))
    plt.figure(figsize=(10, 5))
    plt.bar(x, f1, width=0.4, label='F1 Score')
    plt.xticks(x, valid_labels)
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
