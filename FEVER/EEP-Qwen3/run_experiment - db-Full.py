import os
import json
import sqlite3
import csv
import re
from collections import defaultdict
from model.gpt_verifier import gpt_self_ask_verifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt

from dotenv import load_dotenv
import openai

# === 初始化 API（与上面跑通文件一致）===
load_dotenv()  # 默认读取项目根目录下的 .env
openai.api_key = os.getenv("OPENAI_API_KEY")

# === 路径设置（与上面跑通文件一致）===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                     # ...\FEVER\CoT
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Fever_Data"))    # ...\FEVER\Fever_Data
FEVER_DB_PATH = os.path.join(DATA_DIR, "fever.db")
fever_jsonl   = os.path.join(DATA_DIR, "shared_task_dev.jsonl")
OUTPUT_DIR = BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 300  # 保持和原来一致

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

def extract_answer(output_text: str) -> str:
    if output_text is None:
        return "UNKNOWN"

    s = output_text.strip().upper()
    if s in {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}:
        return s

    pattern = re.compile(
        r'(?:ANSWER|FINAL\s*LABEL|FINAL\s*DECISION|DECISION|FINAL|LABEL|结论|最终结论|最终标签)?\s*[:：]?\s*'
        r'\b(SUPPORTS|REFUTES|NOT\s+ENOUGH\s+INFO)\b',
        flags=re.IGNORECASE
    )
    last = None
    for m in pattern.finditer(output_text):
        last = m
    if last:
        return re.sub(r'\s+', ' ', last.group(1).upper())
    return "UNKNOWN"

def run():
    data = load_fever_data_with_db(fever_jsonl, FEVER_DB_PATH)

    y_true, y_pred = [], []
    file_index = 0
    writer, f = None, None

    for i, (claim, evidence, label) in enumerate(data):
        if i % BATCH_SIZE == 0:
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
            print(f"❌ Error at item {i}: {e}")
            prediction = "UNKNOWN"
            raw_output = str(e)

        y_true.append(label)
        y_pred.append(prediction)
        writer.writerow([claim, evidence, label, prediction, raw_output])

        print(f"{i + 1}")
        print(f"Claim: {claim}\nLabel: {label}\nPredicted: {prediction}\n{'-' * 60}")

    if f:
        f.close()

    print("\nClassification Report:")
    report = classification_report(
        y_true, y_pred, labels=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"], digits=3
    )
    print(report)

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f_report:
        f_report.write(report)

    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

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

    print("✅ Done. Started from scratch, saved and visualized.")

if __name__ == "__main__":
    run()
