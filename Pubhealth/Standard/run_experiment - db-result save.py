import os
import csv
import re
from model.gpt_verifier import gpt_self_ask_verifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt

# === 新增：从项目根目录 .env 加载 API Key ===
from dotenv import load_dotenv
import openai

# 当前脚本目录（.../Pubhealth/CoT）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 项目根目录（.../EEP-CoT-Selfask）
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# 加载 .env
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# 数据目录（.../Pubhealth/Pubhealth_Data）
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Pubhealth_Data"))
PUBHEALTH_TSV = os.path.join(DATA_DIR, "dev.tsv")  # 默认用 dev.tsv

# 输出目录 = 当前 CoT 目录
OUTPUT_DIR = BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 允许的标签
VALID_LABELS = {"TRUE", "FALSE", "MIXTURE", "UNPROVEN"}

# === 提取最终标签 ===
def extract_final_label(output_text: str) -> str:
    if output_text is None:
        return "ERROR"

    text_upper = output_text.strip().upper()

    if text_upper in VALID_LABELS:
        return text_upper

    pattern = re.compile(
        r'(?:FINAL\s*LABEL|FINAL|LABEL|结论|最终结论|最终标签)?\s*[:：]?\s*'
        r'\b(TRUE|FALSE|MIXTURE|UNPROVEN)\b',
        flags=re.IGNORECASE
    )

    last_match = None
    for m in pattern.finditer(output_text):
        last_match = m

    if last_match:
        return last_match.group(1).upper()

    return "ERROR"

# === 加载 PubHealth 数据 ===
def load_pubhealth_data(tsv_path, limit=None):
    data = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            if not row.get("claim") or not row.get("main_text") or not row.get("label"):
                print(f"⚠️ Skipping row {i} due to missing fields.")
                continue
            label = row["label"].upper()
            if label not in VALID_LABELS:
                print(f"⚠️ Skipping row {i} due to invalid label: {label}")
                continue
            claim = row["claim"]
            evidence = row["main_text"]
            data.append((claim, evidence, label))
    return data

def run():
    pubhealth_tsv = PUBHEALTH_TSV
    data = load_pubhealth_data(pubhealth_tsv, limit=None)

    y_true, y_pred = [], []
    batch_size = 100

    # 检查已有的 CSV 文件数量（续跑用）
    file_index = 0
    while os.path.exists(os.path.join(OUTPUT_DIR, f"pubhealth_dev_result_{file_index}.csv")):
        file_index += 1

    start_index = file_index * batch_size
    print(f"🔁 Resuming from file {file_index} / item index {start_index}")
    data = data[start_index:]

    writer, f = None, None
    for i, (claim, evidence, label) in enumerate(data):
        global_index = start_index + i

        # 每 batch_size 开新文件
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
            raw_output = gpt_self_ask_verifier(claim, evidence)
            prediction = extract_final_label(raw_output)
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

    # 输出评估报告
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, digits=3)
    print(report)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f_report:
        f_report.write(report)

    # 绘制 F1 分数图
    labels = sorted(set(y_true + y_pred))
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
