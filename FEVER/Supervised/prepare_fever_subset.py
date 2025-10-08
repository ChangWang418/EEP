#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build FEVER CSVs with real evidence text from local files.
- 路径改为相对 Supervised 的结构
- Fever_Data 中包含 fever.db / train.jsonl / shared_task_dev.jsonl
- 输出到当前目录 data/
"""

import argparse, os, json, random, sqlite3, re
from typing import List, Optional, Tuple, Iterable
import pandas as pd

LABEL_MAP = {
    "SUPPORTS":"SUPPORTS",
    "REFUTES":"REFUTES",
    "NOT ENOUGH INFO":"NEI",
    "NEI":"NEI",
    "true":"SUPPORTS",
    "false":"REFUTES",
    "unknown":"NEI",
}

CLS_LIST = ["SUPPORTS","REFUTES","NEI"]

def norm_label(x:str) -> str:
    return LABEL_MAP.get(str(x), str(x)).upper()

def load_jsonl(path:str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

# ========== sqlite helpers ==========
def open_db(db_path:str):
    """优先只读打开，避免被 DB Browser 占用导致写锁。"""
    if not (db_path and os.path.exists(db_path)):
        raise FileNotFoundError(f"DB not found: {db_path}")
    try:
        return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except Exception:
        return sqlite3.connect(db_path)

def fetch_page_lines(conn, page_id:str) -> Optional[List[str]]:
    try:
        cur = conn.cursor()
        cur.execute("SELECT lines FROM documents WHERE id = ?", (page_id,))
        row = cur.fetchone()
        if not row or row[0] is None:
            return None
        raw = row[0]
        lines = []
        for ln in str(raw).split("\n"):
            if "\t" in ln:  # "idx\ttext"
                _, text = ln.split("\t", 1)
            else:
                text = ln
            text = text.strip()
            if text:
                lines.append(text)
        return lines
    except Exception:
        return None

# ---------- 关键：evidence 解析 ----------
def parse_ev_item(ev_item) -> Optional[Tuple[str,int]]:
    """解析 evidence 项为 (page, sentence_id)。"""
    if not isinstance(ev_item, (list, tuple)) or len(ev_item) < 2:
        return None
    sid = None
    for x in reversed(ev_item):
        if isinstance(x, int):
            sid = x
            break
        if isinstance(x, str) and x.isdigit():
            sid = int(x)
            break
    page = None
    for x in reversed(ev_item):
        if isinstance(x, str) and not x.isdigit():
            page = x.strip()
            break
    if page is None or sid is None:
        return None
    return page, sid

def _title_variants(title: str) -> Iterable[str]:
    """命中失败时的等价标题尝试。"""
    t = title.strip()
    variants = {t}
    t1 = t.split("#", 1)[0]
    variants.update({t1})
    variants.update({t.replace(" ", "_"), t.replace("_", " "),
                     t1.replace(" ", "_"), t1.replace("_", " ")})
    more = set()
    for v in variants:
        more.add(v.replace("–", "-").replace("—", "-"))
    variants |= more
    return list(variants)

def evidence_to_text(conn, ev_field, max_ev_sent:int=5) -> str:
    if not ev_field:
        return ""
    pieces, seen = [], set()
    for group in ev_field:
        if not isinstance(group, (list, tuple)):
            continue
        for ev in group:
            parsed = parse_ev_item(ev)
            if not parsed:
                continue
            page, sid = parsed
            for cand in [page] + _title_variants(page):
                lines = fetch_page_lines(conn, cand)
                if lines is not None and 0 <= sid < len(lines):
                    txt = lines[sid].strip()
                    key = (cand, sid)
                    if key not in seen:
                        seen.add(key)
                        pieces.append(txt)
                    break
            if len(pieces) >= max_ev_sent:
                break
        if len(pieces) >= max_ev_sent:
            break
    text = " ".join(pieces)
    return re.sub(r"\s+", " ", text).strip()

# ========== dataframe builders ==========
def build_df(jsonl_path:str, conn, n_samples:int, seed:int, max_ev_sent:int) -> pd.DataFrame:
    rows = load_jsonl(jsonl_path)
    random.seed(seed)
    random.shuffle(rows)
    if n_samples > 0:
        rows = rows[:min(n_samples, len(rows))]

    claims, evids, labels = [], [], []
    for r in rows:
        claim = r.get("claim","")
        label = norm_label(r.get("label","NEI"))
        ev_field = r.get("evidence", None)
        ev_text = evidence_to_text(conn, ev_field, max_ev_sent=max_ev_sent) if conn else ""
        claims.append(claim)
        evids.append(ev_text)
        labels.append(label)

    df = pd.DataFrame({"claim":claims, "evidence":evids, "label":labels}).fillna("")
    df = df[df["label"].isin(CLS_LIST)]
    return df

# ========== 平衡采样 ==========
def balanced_sample(df: pd.DataFrame, per_class: Optional[int], seed: int) -> pd.DataFrame:
    counts = df["label"].value_counts()
    available = {c: counts.get(c, 0) for c in CLS_LIST}
    if any(available[c] == 0 for c in CLS_LIST):
        missing = [c for c in CLS_LIST if available[c] == 0]
        print(f"[WARN] Missing classes in source data: {missing}. Balanced sampling will drop those classes if absent.")
    vals = [v for v in available.values() if v > 0]
    if not vals:
        print("[WARN] Could not form a balanced set (no class available). Returning original df.")
        return df
    k = min([per_class] + vals) if per_class and per_class > 0 else min(vals)
    parts = []
    for c in CLS_LIST:
        sub = df[df["label"] == c]
        if len(sub) == 0:
            continue
        parts.append(sub.sample(n=min(k, len(sub)), random_state=seed))
    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"[INFO] Balanced sample per class = {k}; total = {len(out)}")
    return out

# ========== main ==========
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # Supervised
    FEVER_DIR = os.path.dirname(BASE_DIR)                      # FEVER
    DATA_DIR  = os.path.join(FEVER_DIR, "Fever_Data")          # Fever_Data
    OUT_DIR   = os.path.join(BASE_DIR, "data")                 # Supervised/data

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=str, default=os.path.join(DATA_DIR, "train.jsonl"))
    ap.add_argument("--dev_jsonl",   type=str, default=os.path.join(DATA_DIR, "shared_task_dev.jsonl"))
    ap.add_argument("--db_path",     type=str, default=os.path.join(DATA_DIR, "fever.db"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--max_ev_sent", type=int, default=5)
    ap.add_argument("--balance", action="store_true", default=True)
    ap.add_argument("--per_class_train", type=int, default=1000)
    ap.add_argument("--per_class_val",   type=int, default=300)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    train_out = os.path.join(OUT_DIR, "train.csv")
    val_out   = os.path.join(OUT_DIR, "val.csv")
    print(f"[INFO] Output dir: {OUT_DIR}")

    conn = None
    if args.db_path:
        try:
            conn = open_db(args.db_path)
            print(f"[OK] Opened DB: {args.db_path}")
        except Exception as e:
            print(f"[WARN] Could not open DB: {e} ; evidence will be empty.")

    df_all = build_df(args.train_jsonl, conn, args.n_samples, args.seed, args.max_ev_sent)

    if os.path.exists(args.dev_jsonl):
        df_val_raw = build_df(args.dev_jsonl, conn, n_samples=0, seed=args.seed+1, max_ev_sent=args.max_ev_sent)
        df_train = df_all
        df_val   = df_val_raw
        if args.balance:
            df_train = balanced_sample(df_train, args.per_class_train, seed=args.seed)
            df_val   = balanced_sample(df_val,   args.per_class_val,   seed=args.seed+1)
    else:
        df_all = df_all.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        n_train = int(len(df_all)*args.train_ratio)
        df_train_raw, df_val_raw = df_all.iloc[:n_train].copy(), df_all.iloc[n_train:].copy()
        if args.balance:
            df_train = balanced_sample(df_train_raw, args.per_class_train, seed=args.seed)
            df_val   = balanced_sample(df_val_raw,   args.per_class_val,   seed=args.seed+1)
        else:
            df_train, df_val = df_train_raw, df_val_raw

    df_train[["claim","evidence","label"]].to_csv(train_out, index=False, encoding="utf-8")
    df_val[["claim","evidence","label"]].to_csv(val_out, index=False, encoding="utf-8")

    if conn is not None:
        conn.close()

    print(f"Saved {train_out} ({len(df_train)})")
    print(f"Saved {val_out} ({len(df_val)})")
    print("[INFO] Train label counts:\n", df_train["label"].value_counts())
    print("[INFO]  Val  label counts:\n", df_val["label"].value_counts())

if __name__ == "__main__":
    main()
