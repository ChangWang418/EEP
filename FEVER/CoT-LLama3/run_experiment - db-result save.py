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

# ========= 新增：API 初始化（按你给的模板） =========
from dotenv import load_dotenv
import openai

load_dotenv()  # By default, load the .env file from the project root directory
openai.api_key = os.getenv("OPENAI_API_KEY")

# ========= 路径：从 CoT 目录指向 Fever_Data =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))              # ...\EEP-CoT-Selfask\FEVER\CoT
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Fever_Data"))  # ...\EEP-CoT-Selfask\FEVER\Fever_Data

FEVER_DB_PATH = os.path.join(DATA_DIR, "fever.db")
fever_jsonl   = os.path.join(DATA_DIR, "shared_task_dev.jsonl")

OUTPUT_DIR = BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 可选健全性检查（路径错时尽早报错）
for p in [FEVER_DB_PATH, fever_jsonl]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Path not found: {p}")
print("DB:", FEVER_DB_PATH)
print("JSONL:", fever_jsonl)

# ========= 自适应版本：兼容不同表/列命名 =========
def get_sentence_from_db(page_title, sentence_index, db_path):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # 常见 FEVER 变体 (表名, id/title列, 正文列)
    candidates = [
        ("documents",  "id",    "lines"),
        ("wikipedia",  "id",    "lines"),
        ("wikipedia",  "title", "text"),
        ("wiki",       "id",    "lines"),
        ("pages",      "id",    "content"),
    ]

    # 兼容 “空格/下划线” 两种标题写法
    tried_titles = [page_title]
    if " " in page_title:
        tried_titles.append(page_title.replace(" ", "_"))
    if "_" in page_title:
        tried_titles.append(page_title.replace("_", " "))

    content = None
    last_err = None

    for table, id_col, text_col in candidates:
        for t in tried_titles:
            try:
                cur.execute(
                    f"SELECT {text_col} FROM {table} WHERE {id_col} = ? LIMIT 1;",
                    (t,)
                )
                row = cur.fetchone()
                if row:
                    content = row[0]
                    break
            except sqlite3.OperationalError as e:
                # 表或列不存在，试下一个候选
                last_err = e
                continue
        if content:
            break

    if not content:
        con.close()
        if last_err:
            return f"[DB schema not matched: last error {last_err}]"
        return f"[Page not found: {page_title}]"

    # bytes 兼容
    if isinstance(content, (bytes, bytearray)):
        content = content.decode("utf-8", errors="ignore")

    # FEVER 常见格式：每行 "index \t sentence"
    for line in str(content).splitlines():
        if "\t" in line:
            idx, sent = line.split("\t", 1)
            try:
                if int(idx) == sentence_index:
                    con.close()
                    return sent
            except ValueError:
                pass

    con.close()
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

# === 从后向前提取标签：仅取最后一次出现（保持原逻辑不变） ===
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

    s = output_text.strip()
    su = s.upper()
    if su in {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}:
        return su

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
    full_data = load_fever_data_with_db(fever_jsonl, FEVER_DB_PATH)

    random.seed(42)
    label_buckets = defaultdict(list)
    for item in full_data:
        label_buckets[item[2]].append(item)

    # 采样规模
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
