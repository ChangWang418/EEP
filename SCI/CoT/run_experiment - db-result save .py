# -*- coding: utf-8 -*-
import os
import json
import csv
import re
import random
from collections import defaultdict
from model.gpt_verifier import gpt_self_ask_verifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt

# ========= 从项目根目录 .env 加载 API Key =========
from dotenv import load_dotenv
import openai

# 项目根目录：.../EEP-CoT-Selfask-Standard-DECOMP
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
openai.api_key = os.getenv("OPENAI_API_KEY")
print("✅ Loaded API key prefix:", (openai.api_key or "")[:10])

# ========= 路径（相对）=========
# 当前脚本目录：.../SCI/CoT
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据目录：.../SCI/Sci_data
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Sci_data"))

# 选择 dev 或 test，这里默认 dev
CLAIMS_PATH = os.path.join(DATA_DIR, "claims_dev.jsonl")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")

# 输出目录：写回 CoT 目录
OUTPUT_DIR = BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= 配置 =========
random.seed(42)
INCLUDE_NOINFOS = True     # True: 三分类（SUPPORTS / REFUTES / NOINFO）
BATCH_SIZE = 500           # 每个CSV批量大小（便于断点续跑）
DEBUG = True               # 是否打印部分原始模型输出
DEBUG_SHOW_N = 20          # 打印前 N 条 raw output
VALID_3 = {"SUPPORTS", "REFUTES", "NOINFO"}

# ========= 抽取器用的正则（宽松，兼容多写法）=========
_PATTERN_MAIN = re.compile(
    r"(?:SO\s+)?THE\s+FINAL\s+ANSWER\s+IS\s*[:：]?\s*"
    r"(SUPPORT|SUPPORTS|REFUTE|REFUTES|CONTRADICT|CONTRADICTS|NO\s*INFO|NOINFO)\b",
    re.IGNORECASE
)
_PATTERN_ALT = re.compile(
    r"(?:FINAL\s+(?:ANSWER|LABEL)|FINAL)\s*[:：]?\s*"
    r"(SUPPORT|SUPPORTS|REFUTE|REFUTES|CONTRADICT|CONTRADICTS|NO\s*INFO|NOINFO)\b",
    re.IGNORECASE
)
_PATTERN_FALLBACK = re.compile(
    r"(SUPPORT|SUPPORTS|REFUTE|REFUTES|CONTRADICT|CONTRADICTS|NO\s*INFO|NOINFO)\b",
    re.IGNORECASE
)

def _norm_token_to_3way(tok: str) -> str:
    """将任意写法规范到：SUPPORTS / REFUTES / NOINFO / UNKNOWN"""
    t = (tok or "").upper().replace(" ", "")
    if t in ("REFUTE", "REFUTES", "CONTRADICT", "CONTRADICTS"):
        return "REFUTES"
    if t in ("SUPPORT", "SUPPORTS"):
        return "SUPPORTS"
    if t in ("NOINFO",):
        return "NOINFO"
    return "UNKNOWN"

def extract_label_from_tail(output_text: str) -> str:
    """
    宽松抽取：从后往前优先匹配
    1) (?:SO )?THE FINAL ANSWER IS [:]? <LABEL>
    2) (FINAL ANSWER|FINAL LABEL|FINAL) [:]? <LABEL>
    3) 兜底：全文最后一次出现的 <LABEL>
    返回：SUPPORTS / REFUTES / NOINFO / UNKNOWN
    """
    if not output_text:
        return "UNKNOWN"
    su = str(output_text).strip()

    # pass-1：主模式
    m_all = list(_PATTERN_MAIN.finditer(su))
    if m_all:
        return _norm_token_to_3way(m_all[-1].group(1))

    # pass-2：备选模式
    a_all = list(_PATTERN_ALT.finditer(su))
    if a_all:
        return _norm_token_to_3way(a_all[-1].group(1))

    # pass-3：兜底（最后一次标签词）
    f_all = list(_PATTERN_FALLBACK.finditer(su))
    if f_all:
        return _norm_token_to_3way(f_all[-1].group(1))

    return "UNKNOWN"

# ========= 标签规范化 =========
def normalize_label(x: str) -> str:
    """把各种写法统一为：SUPPORTS / REFUTES / NOINFO"""
    t = (x or "").strip().upper()
    if t in ["REFUTE", "REFUTES", "CONTRADICT", "CONTRADICTS"]:
        return "REFUTES"
    if t in ["SUPPORT", "SUPPORTS"]:
        return "SUPPORTS"
    if t in ["NOINFO", "NO_INFO", "NEI", "NOT ENOUGH INFO", "NOT-ENOUGH-INFO", "NO INFO"]:
        return "NOINFO"
    return t

# ========= 数据加载（SciFact JSONL）=========
def load_sci_data(claims_path, corpus_path, include_noinfo=True, max_samples=None):
    """
    读取 SciFact 的 claims 与 corpus，返回 (claim, evidence_text, label_raw) 列表。
    - 有标注证据：保留原始标签（support/contradict），评测时统一到 SUPPORTS/REFUTES。
    - 没有证据且 include_noinfo=True：用 cited_doc 的摘要拼接作为 evidence，标签 NOINFO。
    """
    corpus = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            corpus[rec["doc_id"]] = rec

    claims = []
    with open(claims_path, "r", encoding="utf-8") as f:
        for line in f:
            claims.append(json.loads(line))

    final_data = []

    for item in claims:
        claim_text = item.get("claim", "").strip()
        if not claim_text:
            continue

        evidence_map = item.get("evidence", {}) or {}
        cited_ids = item.get("cited_doc_ids", []) or []

        # 1) 收集 SUPPORT/CONTRADICT
        for doc_id_str, evidence_list in evidence_map.items():
            for ev in evidence_list:
                raw_label = ev.get("label", "")
                norm = normalize_label(raw_label)
                if norm not in ["SUPPORTS", "REFUTES"]:
                    continue

                try:
                    doc_id = int(doc_id_str)
                except Exception:
                    try:
                        doc_id = int(str(doc_id_str).strip())
                    except Exception:
                        continue

                doc = corpus.get(doc_id)
                if not doc:
                    continue

                abstract = doc.get("abstract", []) or []
                if not isinstance(abstract, list):
                    abstract = [str(abstract)]

                idxs = ev.get("sentences", []) or []
                evidence_sentences = [abstract[i] for i in idxs if isinstance(i, int) and 0 <= i < len(abstract)]
                evidence_text = " ".join(s.strip() for s in evidence_sentences if s and str(s).strip())
                if not evidence_text:
                    continue

                final_data.append((claim_text, evidence_text, raw_label))

        # 2) 构造 NOINFO：当 evidence 为空时
        if include_noinfo and not evidence_map:
            if cited_ids:
                for pmid in cited_ids:
                    try:
                        doc_id = int(pmid)
                    except Exception:
                        try:
                            doc_id = int(str(pmid).strip())
                        except Exception:
                            continue
                    doc = corpus.get(doc_id)
                    if not doc:
                        continue
                    abstract = doc.get("abstract", []) or []
                    if not isinstance(abstract, list):
                        abstract = [str(abstract)]
                    evidence_text = " ".join(s.strip() for s in abstract if s and str(s).strip())
                    if evidence_text:
                        final_data.append((claim_text, evidence_text, "NOINFO"))

        if max_samples and len(final_data) >= max_samples:
            break

    # 统计（按统一后的三类）
    cnt = defaultdict(int)
    for _, _, lab in final_data:
        cnt[normalize_label(lab)] += 1
    summary = ", ".join([f"{k}:{v}" for k, v in sorted(cnt.items())])
    print(f"✅ Loaded {len(final_data)} samples. Label dist (normalized) -> {summary}")
    return final_data

# ========= 主流程 =========
def run():
    data = load_sci_data(
        CLAIMS_PATH,
        CORPUS_PATH,
        include_noinfo=INCLUDE_NOINFOS,
        max_samples=None
    )

    valid_labels = ["SUPPORTS", "REFUTES", "NOINFO"] if INCLUDE_NOINFOS else ["SUPPORTS", "REFUTES"]

    # 断点续跑：根据已存在文件决定起始批次
    file_index = 0
    while os.path.exists(os.path.join(OUTPUT_DIR, f"sci_dev_result_{file_index}.csv")):
        file_index += 1
    start_index = file_index * BATCH_SIZE
    print(f"🔁 Resuming from file index {file_index} / item index {start_index}")
    data = data[start_index:]

    y_true, y_pred = [], []
    writer, f = None, None

    try:
        for i, (claim, evidence, label_raw) in enumerate(data):
            global_index = start_index + i

            # 每批次新开一个CSV
            if i % BATCH_SIZE == 0:
                if f:
                    f.close()
                csv_path = os.path.join(OUTPUT_DIR, f"sci_dev_result_{file_index}.csv")
                f = open(csv_path, "w", encoding="utf-8", newline="")
                writer = csv.writer(f)
                writer.writerow(["Claim", "Evidence", "True_Label_Raw", "True_Label_Normalized", "Predicted_Label"])
                print(f"\n📁 Writing to {csv_path}")
                file_index += 1

            try:
                raw_pred = gpt_self_ask_verifier(claim, evidence)
                pred_label = extract_label_from_tail(raw_pred)  # SUPPORTS/REFUTES/NOINFO/UNKNOWN

                if DEBUG and i < DEBUG_SHOW_N:
                    print("\n[DEBUG] ----- RAW MODEL OUTPUT START -----")
                    print(raw_pred if isinstance(raw_pred, str) else str(raw_pred))
                    print("[DEBUG] Extracted:", pred_label)
                    print("[DEBUG] ------ RAW MODEL OUTPUT END ------\n")

                if pred_label not in VALID_3:
                    # 匹配失败：三分类下兜底为 NOINFO，二分类下兜底 REFUTES
                    pred_label = "NOINFO" if "NOINFO" in valid_labels else "REFUTES"
            except Exception as e:
                print(f"❌ Error at item {global_index}: {e}")
                pred_label = "NOINFO" if "NOINFO" in valid_labels else "REFUTES"

            true_norm = normalize_label(label_raw)   # 将 support/contradict 统一
            y_true.append(true_norm)
            y_pred.append(pred_label)
            writer.writerow([claim, evidence, label_raw, true_norm, pred_label])

            if DEBUG and i < DEBUG_SHOW_N:
                print(f"{global_index + 1}")
                print(f"Claim: {claim}\nLabel(raw): {label_raw} -> Label(norm): {true_norm}\nPredicted: {pred_label}\n{'-' * 60}")

    finally:
        if f:
            f.close()

    # 评测与可视化
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, labels=valid_labels, digits=3, zero_division=0)
    print(report)

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f_report:
        f_report.write(report)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=valid_labels, zero_division=0
    )

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
    fig_path = os.path.join(OUTPUT_DIR, "f1_score_chart.png")
    plt.savefig(fig_path)
    plt.show()

    print("✅ Done. Saved metrics and chart to:", fig_path)

if __name__ == "__main__":
    run()
