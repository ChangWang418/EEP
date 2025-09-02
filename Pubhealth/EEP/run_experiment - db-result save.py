import os
import csv
import re
from model.gpt_verifier import gpt_self_ask_verifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt

# ===== 新增：从项目根目录 .env 读取 OPENAI_API_KEY =====
from dotenv import load_dotenv
import openai

# 当前脚本目录：.../EEP-CoT-Selfask/Pubhealth/CoT
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 项目根目录：.../EEP-CoT-Selfask
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# 加载根目录 .env 并设置 Key
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# 数据目录：.../EEP-CoT-Selfask/Pubhealth/Pubhealth_Data
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Pubhealth_Data"))

# 选择 dev/test（默认用 dev，如需 test 改成 "test.tsv"）
PUBHEALTH_TSV = os.path.join(DATA_DIR, "dev.tsv")

# 输出目录：放在脚本所在的 CoT 目录
OUTPUT_DIR = BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 标签与抽取逻辑（保持原有逻辑）=====
VALID_LABELS = {"TRUE", "FALSE", "UNPROVEN", "MIXTURE"}

# === 从后向前提取标签（仅取最后一次出现的标签）===
def extract_final_label_from_tail(text: str) -> str:
    """
    从模型输出中只取 Step 4/Final label 的标签。
    规则：从后向前查找，取最后一个 (可带 'Final label') 的标签匹配。
    匹配失败 -> 抛出 ValueError（外层捕获并记为 ERROR）。
    """
    if text is None:
        raise ValueError("empty model output")

    pattern = re.compile(
        r'(?:final\s*label|final|label|结论|最终结论|最终标签)?\s*[:：]?\s*'
        r'\b(TRUE|FALSE|UNPROVEN|MIXTURE)\b',
        flags=re.IGNORECASE
    )

    last_match = None
    for m in pattern.finditer(text):
        last_match = m

    if not last_match:
        raise ValueError("no final label found")

    label = last_match.group(1).upper()
    if label not in VALID_LABELS:
        raise ValueError(f"invalid label parsed: {label}")

    return label

# 加载 PubHealth 的 tsv 数据
def load_pubhealth_data(tsv_path, limit=None):
    data = []
    valid_labels = VALID_LABELS
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
                continue
            claim = row["claim"]
            evidence = row["main_text"]
            data.append((claim, evidence, label))
    return data

def run():
    # 使用相对路径的数据
    pubhealth_tsv = PUBHEALTH_TSV
    data = load_pubhealth_data(pubhealth_tsv, limit=None)

    y_true, y_pred = [], []
    batch_size = 100

    # 续跑：根据已有文件数量继续
    file_index = 0
    while os.path.exists(os.path.join(OUTPUT_DIR, f"pubhealth_dev_result_{file_index}.csv")):
        file_index += 1

    start_index = file_index * batch_size
    print(f"🔁 Resuming from file {file_index} / item index {start_index}")
    data = data[start_index:]

    writer, f = None, None

    for i, (claim, evidence, label) in enumerate(data):
        global_index = start_index + i

        # 每 batch_size 开一个新文件
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
            prediction = extract_final_label_from_tail(raw_output)
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

    # 评估报告与图表（保持原逻辑）
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, digits=3)
    print(report)

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f_report:
        f_report.write(report)

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
