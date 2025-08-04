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

load_dotenv()  # By default, load the .env file from the project root directory
openai.api_key = os.getenv("OPENAI_API_KEY")

# Current file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Point to the Fever_Data folder
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Fever_Data"))

# Paths to database and JSON files
FEVER_DB_PATH = os.path.join(DATA_DIR, "fever.db")
fever_jsonl = os.path.join(DATA_DIR, "shared_task_dev.jsonl")

# Output directory (can be the same as DATA_DIR or a sibling directory)
OUTPUT_DIR = BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create if it doesn't exist

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

def load_fever_data_with_db(jsonl_path, db_path):
    data = []
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
            data.append((claim, evidence_text, label))
    return data

def extract_answer(text):
    match = re.search(r'(ANSWER[:：]?)?\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO)', text.strip(), re.IGNORECASE)
    if match:
        return match.group(2).upper()
    return "UNKNOWN"

def run():
    full_data = load_fever_data_with_db(fever_jsonl, FEVER_DB_PATH)

    random.seed(42)
    label_buckets = defaultdict(list)
    for item in full_data:
        label_buckets[item[2]].append(item)

    sample_size = 666
    sampled_data = []
    for label in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
        bucket = label_buckets[label]
        if len(bucket) < sample_size:
            raise ValueError(f"❌ Not enough samples for label '{label}': only {len(bucket)} found.")
        sampled = random.sample(bucket, sample_size)
        sampled_data.extend(sampled)

    random.shuffle(sampled_data)
    data = sampled_data

    y_true, y_pred = [], []
    batch_size = 500
    file_index = 0
    while os.path.exists(os.path.join(OUTPUT_DIR, f"fever_dev_result_{file_index}.csv")):
        file_index += 1

    start_index = 0
    writer, f = None, None

    for i, (claim, evidence, label) in enumerate(data):
        global_index = start_index + i
        if i % batch_size == 0:
            if f:
                f.close()
            csv_path = os.path.join(OUTPUT_DIR, f"fever_dev_result_{file_index}.csv")
            f = open(csv_path, "w", encoding="utf-8", newline="")
            writer = csv.writer(f)
            writer.writerow(["Claim", "Evidence", "True_Label", "Predicted_Label", "Raw_Output"])
            print(f"\n📁 Writing to {csv_path}")
            file_index += 1

        try:
            raw_output = gpt_self_ask_verifier(claim, evidence)
            prediction = extract_answer(raw_output)
        except Exception as e:
            print(f"❌ Error at item {global_index}: {e}")
            prediction = "ERROR"
            raw_output = str(e)

        y_true.append(label)
        y_pred.append(prediction)
        writer.writerow([claim, evidence, label, prediction, raw_output])

        print(f"{global_index + 1}")
        print(f"Claim: {claim}\nLabel: {label}\nPredicted: {prediction}\n{'-' * 60}")

    if f:
        f.close()

    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, labels=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"], digits=3)
    print(report)

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f_report:
        f_report.write(report)

    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
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
