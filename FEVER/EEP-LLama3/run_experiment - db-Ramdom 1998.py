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

# === 环境与 OpenAI API（从 .env 读取）===
from dotenv import load_dotenv
import openai
load_dotenv()  # 默认从项目根目录的 .env 读取
openai.api_key = os.getenv("OPENAI_API_KEY")

# 固定随机种子（保持原逻辑）
random.seed(42)

# === 路径（从当前脚本 CoT 目录指向上一级 Fever_Data）===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                  # ...\FEVER\CoT
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Fever_Data")) # ...\FEVER\Fever_Data

# 数据库与 JSONL 路径（与已跑通文件保持一致：从 Fever_Data 读取）
FEVER_DB_PATH = os.path.join(DATA_DIR, "fever.db")
fever_jsonl   = os.path.join(DATA_DIR, "shared_task_dev.jsonl")

# 输出目录：使用脚本所在目录
OUTPUT_DIR = BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# === 从后向前提取标签：仅取最后一次出现（保留原逻辑） ===
def extract_answer(output_text: str) -> str:
    """
    从模型输出中“从后向前”提取最终答案标签（仅取最后一次出现）。
    允许格式：
      - Answer: SUPPORTS
      - Final label：REFUTES
      - Final decision: NOT ENOUGH INFO
      - Label: NOT ENOUGH INFO
      - 仅有标签本身（如 SUPPORTS）
    匹配不到返回 "UNKNOWN"。
    """
    if output_text is None:
        return "UNKNOWN"

    # 快速路径：整体就是一个标签
    s = output_text.strip()
    su = s.upper()
    if su in {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}:
        return su

    # 正则：可选前缀 + 三类标签；使用 finditer 并取最后一个匹配
    pattern = re.compile(
        r'(?:ANSWER|FINAL\s*LABEL|FINAL\s*DECISION|DECISION|FINAL|LABEL|结论|最终结论|最终标签)?\s*[:：]?\s*'
        r'\b(SUPPORTS|REFUTES|NOT\s+ENOUGH\s+INFO)\b',
        flags=re.IGNORECASE
    )

    last = None
    for m in pattern.finditer(output_text):
        last = m

    if last:
        return re.sub(r'\s+', ' ', last.group(1).upper())  # 规范为单空格

    return "UNKNOWN"

def run():
    # 使用新的全局路径变量（不改原有逻辑）
    full_data = load_fever_data_with_db(fever_jsonl, FEVER_DB_PATH)

    random.seed(42)
    label_buckets = defaultdict(list)
    for item in full_data:
        label_buckets[item[2]].append(item)

    # 采样规模（保持原值 666）
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
    batch_size = 300  # 保持原逻辑
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
            prediction = "UNKNOWN"
            raw_output = str(e)

        y_true.append(label)
        y_pred.append(prediction)
        writer.writerow([claim, evidence, label, prediction, raw_output])
    
        print(f"{global_index + 1}")
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

    print("✅ Done. All resumed, saved and visualized.")

if __name__ == "__main__":
    run()
