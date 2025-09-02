import os
import json
import sqlite3
import csv
import random
import re
from collections import defaultdict
from model.gpt_verifier import gpt_self_ask_verifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt

# === 按要求：API 初始化（从 .env 读取 OPENAI_API_KEY）===
from dotenv import load_dotenv
import openai
load_dotenv()  # 默认从项目根目录的 .env 读取
openai.api_key = os.getenv("OPENAI_API_KEY")

random.seed(42)

# === 按要求：相对路径（从当前脚本所在 CoT 目录指向上一级 Fever_Data）===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                 # ...\FEVER\CoT
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Fever_Data"))  # ...\FEVER\Fever_Data

# 数据库与 JSONL 路径（与上面那份一致：都从 Fever_Data 读取）
FEVER_DB_PATH = os.path.join(DATA_DIR, "fever.db")
fever_jsonl   = os.path.join(DATA_DIR, "shared_task_dev.jsonl")

# 输出目录：使用脚本所在目录（CoT）
OUTPUT_DIR = BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 1) 只允许的标签（FEVER 三分类） =====
VALID_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

# ===== 2) 提取最终标签的函数（从后向前，仅取最后一次） =====
def extract_final_label(output_text: str) -> str:
    """
    从 LLM 的长输出中提取最终标签。
    只取“最后一次出现”的标签（从后向前），以规避中间推理中的中间结论干扰。
    支持格式：
      - "FINAL LABEL: <LABEL>"
      - "FINAL DECISION: <LABEL>"
      - "ANSWER: <LABEL>"
      - "LABEL: <LABEL>"
      - "结论/最终结论/最终标签: <LABEL>"
      - 仅标签本身（如 "NOT ENOUGH INFO"）
    匹配不到返回 "NO_LABEL"。
    """
    if not output_text:
        return "NO_LABEL"

    s = output_text.strip()
    su = s.upper()

    # 快速路径：整体即为一个标签
    if su in {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}:
        return su

    # 可选前缀 + 标签；使用 finditer 并取“最后一个”匹配
    pattern = re.compile(
        r'(?:ANSWER|FINAL\s*LABEL|FINAL\s*DECISION|DECISION|FINAL|LABEL|结论|最终结论|最终标签)?\s*[:：]?\s*'
        r'\b(SUPPORTS|REFUTES|NOT\s+ENOUGH\s+INFO)\b',
        flags=re.IGNORECASE
    )

    last = None
    for m in pattern.finditer(output_text):
        last = m

    if last:
        # 统一空格（NOT ENOUGH INFO -> 保持单空格）
        return re.sub(r'\s+', ' ', last.group(1).upper())

    return "NO_LABEL"

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
                evidence_text = " ".join(sentences) if sentences else "[No evidence extracted]"
            else:
                evidence_text = "[No evidence provided]"

            # 只收集有效标签样本
            if label in VALID_LABELS:
                data_by_label[label].append((claim, evidence_text, label))

    # 每类采样
    final_data = []
    for label, items in data_by_label.items():
        if per_class_limit is None:
            sampled = items
        else:
            sampled = random.sample(items, min(len(items), per_class_limit))
        final_data.extend(sampled)

    random.shuffle(final_data)
    return final_data

def run(debug_print=False):
    # === 仅更改路径与 API 载入方式，其余逻辑保持不变 ===
    data = load_fever_data_with_db(fever_jsonl, FEVER_DB_PATH, per_class_limit=666)

    y_true, y_pred = [], []
    batch_size = 500

    # 续跑：从已存在的文件序号 * batch_size 继续
    file_index = 0
    while os.path.exists(os.path.join(OUTPUT_DIR, f"fever_dev_result_{file_index}.csv")):
        file_index += 1

    start_index = file_index * batch_size
    print(f"🔁 Resuming from file {file_index} / item index {start_index}")
    data = data[start_index:]

    writer, f = None, None

    for i, (claim, evidence, label) in enumerate(data):
        global_index = start_index + i

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
            if debug_print:
                print("===== RAW MODEL OUTPUT BEGIN =====")
                print(raw_output)
                print("===== RAW MODEL OUTPUT END =====")
            prediction = extract_final_label(raw_output)
        except Exception as e:
            print(f"❌ Error at item {global_index}: {e}")
            prediction = "ERROR"

        # 非法/缺失标签统一为 NO_LABEL，便于统计
        if prediction not in VALID_LABELS:
            prediction = "NO_LABEL"

        y_true.append(label)
        y_pred.append(prediction)
        writer.writerow([claim, evidence, label, prediction])

        print(f"{global_index + 1}")
        print(f"Claim: {claim}\nLabel: {label}\nPredicted: {prediction}\n{'-' * 60}")

    if f:
        f.close()

    # 只对有效预测做评估
    eval_true = []
    eval_pred = []
    for yt, yp in zip(y_true, y_pred):
        if yp in VALID_LABELS:
            eval_true.append(yt)
            eval_pred.append(yp)

    print("\nClassification Report (Only valid predictions):")
    print(classification_report(eval_true, eval_pred, labels=VALID_LABELS, digits=3))

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f_report:
        f_report.write(classification_report(eval_true, eval_pred, labels=VALID_LABELS, digits=3))

    precision, recall, f1, _ = precision_recall_fscore_support(
        eval_true, eval_pred, labels=VALID_LABELS, zero_division=0
    )

    x = range(len(VALID_LABELS))
    plt.figure(figsize=(10, 5))
    plt.bar(x, f1, width=0.4, label='F1 Score')
    plt.xticks(x, VALID_LABELS)
    plt.ylabel("F1 Score")
    plt.title("F1 Score per Label (Valid Predictions Only)")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "f1_score_chart.png"))
    plt.show()

    print("✅ Done. All saved and visualized.")

if __name__ == "__main__":
    # debug_print=True 可打印模型原始输出，便于排查
    run(debug_print=False)
