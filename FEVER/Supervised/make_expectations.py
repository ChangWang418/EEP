#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Make expectations for EEP supervised training.
Input : Fever_Data/train.csv (claim,evidence,label), Fever_Data/val.csv
Output: Supervised/data/expectations_train.jsonl, Supervised/data/expectations_val.jsonl
Each line: {"idx": i, "claim": "...", "pos_exp": [...], "neg_exp": [...]}

改动要点：
- 更细的句子/子句切分（即使 evidence 只有 1 句，也能切出多个子句）
- 新增 mode="hybrid"：优先从 evidence 取 1-2 条“像事实”的句子，再用 claim 短语补齐到 k
- 增加随机种子，保证可复现
- 路径相对化：脚本位于 Supervised，数据位于其上级目录 FEVER 下的 Fever_Data
"""

import os, argparse, json, re, random
import pandas as pd

# ====== 路径锚点 ======
# 当前脚本所在目录（Supervised）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 上一级目录（FEVER）
FEVER_DIR = os.path.dirname(BASE_DIR)

# ========= 句子/子句切分 =========
def sent_split(text: str):
    """先按句末标点切句，再按逗号/冒号/破折号/中文标点切子句；最多 8 段。"""
    text = str(text).replace("\n", " ").strip()
    if not text:
        return []
    # 先按句末标点
    rough = re.split(r'(?<=[\.\?\!;；。！？])\s+', text)
    segs = []
    for p in rough:
        p = p.strip()
        if not p:
            continue
        # 再按子句：逗号、冒号、分号、破折号、顿号等
        subs = re.split(r'\s*[,，:：;；—\-、]+\s*', p)
        for s in subs:
            s = s.strip()
            if s:
                segs.append(s)
    return segs[:8]

# ========= 从 claim 生成短语 =========
def phrases_from_claim(claim: str, k=3):
    """
    按标点/连词切成短语，保留 4~14 词的片段，去重后取前 k 条。
    （较原版放宽，命中率更高）
    """
    parts = re.split(r'[,:;–—\-()]+|\band\b|\bor\b|\bbut\b', claim, flags=re.I)
    phrases = []
    for p in parts:
        toks = p.strip().split()
        if 4 <= len(toks) <= 14:
            phrases.append(" ".join(toks))
    if not phrases:
        phrases = [claim]
    seen = set(); uniq = []
    for x in phrases:
        key = x.lower().strip()
        if key and key not in seen:
            uniq.append(x); seen.add(key)
    return uniq[:k]

# ========= 简单否定生成 =========
def negate(phrase: str):
    """在助动词后插入 not；若没有助动词则前置“not ”（够用的 baseline，可后续换 LLM）。"""
    toks = phrase.split()
    for i, w in enumerate(toks):
        lw = w.lower()
        if lw in ["is","are","was","were","has","have","do","does","did",
                  "can","could","should","would","will","may","might"]:
            return " ".join(toks[:i+1] + ["not"] + toks[i+1:])
    return "not " + phrase

# ========= 核心：构造期望 =========
def build_expectations(row, mode="hybrid", k=3):
    claim = str(row["claim"])
    evidence = str(row.get("evidence", ""))

    if mode == "heuristic":
        pos = phrases_from_claim(claim, k=k)
        neg = [negate(x) for x in pos]

    elif mode == "from_evidence":
        sents = sent_split(evidence)
        # 更像“事实”的句子：含系动词/年份/典型谓词
        cand = [s for s in sents if re.search(r'\b(is|was|were|are|has|have|born|served|founded|located)\b', s, re.I)
                               or re.search(r'\b\d{3,4}\b', s)]
        if not cand:
            cand = sents
        pos = cand[:max(k, 1)] if cand else phrases_from_claim(claim, k=k)
        neg = [negate(x) for x in pos]

    elif mode == "hybrid":
        # 先从 evidence 取 1-2 条“像事实”的句子，再用 claim 短语补齐到 k
        sents = sent_split(evidence)
        facty = [s for s in sents if re.search(r'\b(is|was|were|are|has|have|born|served|founded|located)\b', s, re.I)
                                 or re.search(r'\b\d{3,4}\b', s)]
        pos = (facty[:2] if facty else [])
        if len(pos) < k:
            extra = [x for x in phrases_from_claim(claim, k=k) if x not in pos]
            pos = (pos + extra)[:k]
        neg = [negate(x) for x in pos]

    else:
        raise ValueError("mode must be heuristic|from_evidence|hybrid")

    # 去空、去重、截断
    def _clean(lst):
        out, seen = [], set()
        for x in lst:
            t = x.strip()
            if not t:
                continue
            key = t.lower()
            if key not in seen:
                seen.add(key); out.append(t)
        return out[:k]

    pos = _clean(pos)
    neg = _clean(neg)
    return pos, neg

# ========= 跑一个 split =========
def run_split(csv_path, out_path, mode="hybrid", k=3):
    df = pd.read_csv(csv_path)
    items = []
    for i, row in df.iterrows():
        pos, neg = build_expectations(row, mode=mode, k=k)
        items.append({
            "idx": int(i),
            "claim": str(row["claim"]),
            "pos_exp": pos,
            "neg_exp": neg
        })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {out_path} ({len(items)})")

# ========= main =========
def main():
    ap = argparse.ArgumentParser()
    # 读取 Supervised/data 下的 CSV（不是 Fever_Data）
    ap.add_argument("--train_csv", type=str, default=os.path.join(BASE_DIR, "data", "train.csv"))
    ap.add_argument("--val_csv",   type=str, default=os.path.join(BASE_DIR, "data", "val.csv"))
    # 输出到同一目录下
    ap.add_argument("--out_train", type=str, default=os.path.join(BASE_DIR, "data", "expectations_train.jsonl"))
    ap.add_argument("--out_val",   type=str, default=os.path.join(BASE_DIR, "data", "expectations_val.jsonl"))
    ap.add_argument("--mode", type=str, default="hybrid",
                    choices=["heuristic", "from_evidence", "hybrid"])
    ap.add_argument("--k", type=int, default=3, help="每条 claim 的正/反期望数量上限")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    run_split(args.train_csv, args.out_train, mode=args.mode, k=args.k)
    run_split(args.val_csv,   args.out_val,   mode=args.mode, k=args.k)

if __name__ == "__main__":
    main()
