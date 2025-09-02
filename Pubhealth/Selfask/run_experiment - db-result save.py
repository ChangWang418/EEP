import os
import csv
import re
from model.gpt_verifier import gpt_self_ask_verifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt

# === 新增：从项目根目录 .env 读取 API Key ===
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

# 默认使用 dev.tsv，如需切换到 test 改这里
PUBHEALTH_TSV = os.path.join(DATA_DIR, "dev.tsv")

# 输出目录：放在当前 CoT 目录
OUTPUT_DIR = BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 标签与映射 =====
VALID_LABELS = {"TRUE", "FALSE", "UNPROVEN", "MIXTURE"}
ALIAS_MAP = {
    "SUPPORT": "TRUE",
    "SUPPORTS": "TRUE",
    "REFUTE": "FALSE",
    "REFUTES": "FALSE",
    "YES": "TRUE",
    "NO": "FALSE",
}

def _normalize_label(raw: str) -> str:
    if not raw:
        return "ERROR"
    raw_up = re.sub(r"\s+", " ", raw.strip().upper())
    return ALIAS_MAP.get(raw_up, raw_up if raw_up in VALID_LABELS else "ERROR")

# === 从后向前提取标签 ===
def extract_final_label_from_tail(output_text: str) -> str:
    if not output_text:
        return "ERROR"
    su = output_text.upper().strip()

    # 1) 优先匹配结论句
    fa_pattern = re.compile(
        r"SO\s+THE\s+FINAL\s+ANSWER\s+IS\s*[:：]?\s*"
        r"(TRUE|FALSE|UNPROVEN|MIXTURE|SUPPORTS?|REFUTES?|YES|NO)"
        r"(?:\s*(?:[\.\!\?，。；;：:]|$))",
        flags=re.IGNORECASE
    )
    m1 = list(fa_pattern.finditer(su))
    if m1:
        return _normalize_label(m1[-1].group(1))

    # 2) 兜底：取最后一个独立标签/别名
    tail_pattern = re.compile(
        r"(TRUE|FALSE|UNPROVEN|MIXTURE|SUPPORTS?|REFUTES?|YES|NO)"
        r"(?:\s*(?:[\.\!\?，。；;：:]|$))",
        flags=re.IGNORECASE
    )
    m2 = list(tail_pattern.finditer(su))
    if m2:
        return _normalize_label(m2[-1].group(1))

    return "ERROR"

# ===== 加载 PubHealth 的 tsv 数据 =====
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

    # 检查已有的 CSV 文件数量（续跑）
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
            writer.writerow(["Claim", "Evidence", "True_Label", "Predicted_Label", "Raw_Output"])
            print(f"\n📁 Writing to {csv_path}")
            file_index += 1

        try:
            raw_output = gpt_self_ask_verifier(claim, evidence)
            prediction = extract_final_label_from_tail(raw_output)
            if prediction not in VALID_LABELS:
                prediction = "ERROR"
        except Exception as e:
            print(f"❌ Error at item {global_index}: {e}")
            prediction, raw_output = "ERROR", str(e)

        y_true.append(label)
        y_pred.append(prediction)
        writer.writerow([claim, evidence, label, prediction, raw_output])

        print(f"{global_index + 1}: {prediction}")

    if f:
        f.close()

    # 输出评估报告
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, labels=list(VALID_LABELS), digits=3)
    print(report)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f_report:
        f_report.write(report)

    # 绘制 F1 分数图
    labels_for_plot = sorted(set(y_true + y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_for_plot, zero_division=0
    )
    x = range(len(labels_for_plot))
    plt.figure(figsize=(10, 5))
    plt.bar(x, f1, width=0.4, label='F1 Score')
    plt.xticks(x, labels_for_plot)
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
