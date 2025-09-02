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

# ========= 从项目根目录 .env 读取 OPENAI_API_KEY =========
from dotenv import load_dotenv
import openai

# 假设脚本位于 .../SCIdecompose/CoT/ 目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                 # .../SCIdecompose/CoT
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))    # .../SCIdecompose
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
openai.api_key = os.getenv("OPENAI_API_KEY")
print("✅ API key prefix:", (openai.api_key or "")[:10])

random.seed(42)

# ========= 相对路径（保持你原来的文件名）=========
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Sci_data"))  # .../SCIdecompose/Sci_data
CLAIMS_PATH = os.path.join(DATA_DIR, "claims_dev.jsonl")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")
OUTPUT_DIR  = BASE_DIR                                                # 输出到当前 CoT 目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== 可配置开关 ======
INCLUDE_NOINFOS = True
BATCH_SIZE = 50
DEBUG = True
DEBUG_SHOW_N = 20000

# ====== 标签空间 ======
VALID_3 = {"SUPPORTS", "REFUTES", "NOINFO"}

# ====== 正则 ======
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
    t = (tok or "").upper().replace(" ", "")
    if t in ("REFUTE", "REFUTES", "CONTRADICT", "CONTRADICTS"):
        return "REFUTES"
    if t in ("SUPPORT", "SUPPORTS"):
        return "SUPPORTS"
    if t in ("NOINFO",):
        return "NOINFO"
    return "UNKNOWN"

def extract_label_from_tail(output_text: str) -> str:
    if not output_text:
        return "UNKNOWN"
    su = str(output_text).strip()
    m_all = list(_PATTERN_MAIN.finditer(su))
    if m_all:
        return _norm_token_to_3way(m_all[-1].group(1))
    a_all = list(_PATTERN_ALT.finditer(su))
    if a_all:
        return _norm_token_to_3way(a_all[-1].group(1))
    f_all = list(_PATTERN_FALLBACK.finditer(su))
    if f_all:
        return _norm_token_to_3way(f_all[-1].group(1))
    return "UNKNOWN"

def normalize_label(x: str) -> str:
    t = (x or "").strip().upper()
    if t in ["REFUTE", "REFUTES", "CONTRADICT", "CONTRADICTS"]:
        return "REFUTES"
    if t in ["SUPPORT", "SUPPORTS"]:
        return "SUPPORTS"
    if t in ["NOINFO", "NO_INFO", "NEI", "NOT ENOUGH INFO", "NOT-ENOUGH-INFO", "NO INFO"]:
        return "NOINFO"
    return t

def load_sci_data(claims_path, corpus_path, include_noinfo=True, max_samples=None):
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

        for doc_id_str, evidence_list in evidence_map.items():
            for ev in evidence_list:
                raw_label = ev.get("label", "")
                if normalize_label(raw_label) not in ["SUPPORTS", "REFUTES"]:
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

    cnt = defaultdict(int)
    for _, _, lab in final_data:
        cnt[normalize_label(lab)] += 1
    summary = ", ".join([f"{k}:{v}" for k, v in sorted(cnt.items())])
    print(f"✅ Loaded {len(final_data)} samples. Label dist (normalized) -> {summary}")
    return final_data

def run():
    data = load_sci_data(CLAIMS_PATH, CORPUS_PATH, include_noinfo=INCLUDE_NOINFOS, max_samples=None)
    valid_labels = ["SUPPORTS", "REFUTES", "NOINFO"] if INCLUDE_NOINFOS else ["SUPPORTS", "REFUTES"]

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
                pred_label = extract_label_from_tail(raw_pred)
                if DEBUG and i < DEBUG_SHOW_N:
                    print("\n[DEBUG] ----- RAW MODEL OUTPUT START -----")
                    print(raw_pred if isinstance(raw_pred, str) else str(raw_pred))
                    print("[DEBUG] Extracted:", pred_label)
                    print("[DEBUG] ------ RAW MODEL OUTPUT END ------\n")
                if pred_label not in VALID_3:
                    pred_label = "NOINFO" if "NOINFO" in valid_labels else "REFUTES"
            except Exception as e:
                print(f"❌ Error at item {global_index}: {e}")
                pred_label = "NOINFO" if "NOINFO" in valid_labels else "REFUTES"

            true_norm = normalize_label(label_raw)
            y_true.append(true_norm)
            y_pred.append(pred_label)
            writer.writerow([claim, evidence, label_raw, true_norm, pred_label])

            if DEBUG and i < DEBUG_SHOW_N:
                print(f"{global_index + 1}")
                print(f"Claim: {claim}\nLabel(raw): {label_raw} -> Label(norm): {true_norm}\nPredicted: {pred_label}\n{'-' * 60}")
    finally:
        if f:
            f.close()

    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, labels=valid_labels, digits=3, zero_division=0)
    print(report)

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f_report:
        f_report.write(report)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=valid_labels, zero_division=0)
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
