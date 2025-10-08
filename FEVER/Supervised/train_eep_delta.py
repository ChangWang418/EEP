#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervised EEP training with S/R/Other logits on top of a delta scorer.
- delta: cosine similarity of sentence embeddings (frozen by default)
- trainable params: scale/bias for logits + gamma; optional encoder finetune
- pooling: max / lse (soft) / topk
Inputs :
  data/train.csv, data/val.csv
  data/expectations_train.jsonl, data/expectations_val.jsonl
Output:
  outputs/eep_supervised/ (best.pt, metrics.json, config.json[, val_preds.csv])
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os, argparse, json, math, re, random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

# ==== 根路径：当前脚本所在目录（FEVER/Supervised） ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def simple_sent_split(text):
    text = str(text).replace("\n", " ")
    parts = re.split(r'(?<=[\.\?\!;])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts[:8] if parts else [""]

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class DeltaEncoder(nn.Module):
    """句向量编码器：默认使用 sentence-transformers 模型（transformers backend）"""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", finetune=False):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.enc = AutoModel.from_pretrained(model_name)
        self.finetune = finetune
        if not finetune:
            for p in self.enc.parameters():
                p.requires_grad = False

    def mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts

    def encode(self, texts):
        batch = self.tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        out = self.enc(**batch)
        emb = self.mean_pool(out.last_hidden_state, batch["attention_mask"])
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb  # [B, d]

class EEPHead(nn.Module):
    """把 S/R 转为三类 logits，并含 trainable gamma/scale/bias/temperature"""
    def __init__(self, init_gamma=1.0, freeze_gamma=False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(3))
        self.bias  = nn.Parameter(torch.zeros(3))
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)), requires_grad=not freeze_gamma)
        self.temp  = nn.Parameter(torch.tensor(1.0))

    def forward(self, S, R, M):
        """
        S,R,M: [B]，其中 M 是与聚合方式一致的 max/soft 聚合结果
        O = gamma * (1 - M)
        """
        O = self.gamma * (1.0 - M)
        logits = torch.stack([S, R, O], dim=-1)  # [B,3]
        logits = logits * self.scale + self.bias
        logits = logits / self.temp.clamp(min=1e-3)
        return logits, O

def aggregate_pairwise_sims(sims, mode="max", beta=10.0, topk=3):
    """
    sims: [m, n] = pos/neg embeddings 与 evidence embeddings 的余弦相似度
    返回标量聚合分数：
      - max:  max_{i,j} sims[i,j]
      - lse:  (1/beta)*logsumexp(beta*sims)
      - topk: 扁平化后取前 k 个再取最大
    """
    if sims.numel() == 0:
        return torch.tensor(0.0, device=sims.device)

    if mode == "max":
        return sims.max()
    elif mode == "lse":
        x = sims.view(-1)
        return (torch.logsumexp(beta * x, dim=0) / beta)
    elif mode == "topk":
        x = sims.view(-1)
        k = min(topk, x.numel())
        vals, _ = torch.topk(x, k)
        return vals.max()
    else:
        raise ValueError(f"Unknown pooling mode: {mode}")

def compute_SR_M(encoder, pos_list, neg_list, ev_sents, pooling="max", beta=10.0, topk=3,
                  k_pos=None, k_neg=None, rng=None):
    """返回：S, R, M"""
    if not ev_sents:
        ev_sents = [""]

    # 期望随机下采样
    if k_pos is not None and pos_list and len(pos_list) > k_pos:
        rng = rng or random
        pos_list = rng.sample(pos_list, k_pos)
    if k_neg is not None and neg_list and len(neg_list) > k_neg:
        rng = rng or random
        neg_list = rng.sample(neg_list, k_neg)

    ev_emb  = encoder.encode(ev_sents)
    S = torch.tensor(0.0, device=ev_emb.device)
    R = torch.tensor(0.0, device=ev_emb.device)

    if pos_list:
        pos_emb = encoder.encode(pos_list)
        sims_pos = pos_emb @ ev_emb.T
        S = aggregate_pairwise_sims(sims_pos, mode=pooling, beta=beta, topk=topk)

    if neg_list:
        neg_emb = encoder.encode(neg_list)
        sims_neg = neg_emb @ ev_emb.T
        R = aggregate_pairwise_sims(sims_neg, mode=pooling, beta=beta, topk=topk)

    if pooling in ("max", "topk"):
        M = torch.maximum(S, R)
    else:
        both = []
        if pos_list:
            both.append(sims_pos.view(-1))
        if neg_list:
            both.append(sims_neg.view(-1))
        if both:
            all_flat = torch.cat(both, dim=0)
            M = (torch.logsumexp(beta * all_flat, dim=0) / beta)
        else:
            M = torch.tensor(0.0, device=ev_emb.device)

    return S, R, M

def build_dataset(csv_path, exp_path):
    df = pd.read_csv(csv_path)
    exps = load_jsonl(exp_path) if os.path.exists(exp_path) else []
    idx2exp = {int(x.get("idx", i)): x for i, x in enumerate(exps)}
    items = []
    for i, row in df.iterrows():
        ex = idx2exp.get(int(i))
        if ex is None:
            ex = {"pos_exp":[str(row["claim"])], "neg_exp":[f"not {row['claim']}"]}
        items.append({
            "claim": str(row["claim"]),
            "evidence": str(row.get("evidence","")),
            "label": str(row["label"]).strip().upper(),
            "pos_exp": ex.get("pos_exp", []),
            "neg_exp": ex.get("neg_exp", []),
        })
    return items

def label_to_idx(lbl):
    if "SUPPORT" in lbl: return 0
    if "REFUTE"  in lbl: return 1
    return 2  # NEI

def evaluate(encoder, head, data, pooling, beta, topk, margin_m=0.0, lambda_reg=0.0,
             k_pos=None, k_neg=None, seed=42, dump_path=None):
    encoder.eval(); head.eval()
    rng = random.Random(seed)
    y_true, y_pred = [], []
    loss_ce_total = 0.0; n = 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    rows = []
    for it in data:
        ev_sents = simple_sent_split(it["evidence"])
        S, R, M = compute_SR_M(
            encoder, it["pos_exp"], it["neg_exp"], ev_sents,
            pooling=pooling, beta=beta, topk=topk, k_pos=k_pos, k_neg=k_neg, rng=rng
        )
        logits, O = head(S.unsqueeze(0), R.unsqueeze(0), M.unsqueeze(0))  # [1,3]
        y = torch.tensor([label_to_idx(it["label"])], device=device)
        loss = ce(logits, y)
        if margin_m>0 and lambda_reg>0:
            reg = torch.clamp(margin_m - torch.abs(S - R), min=0.0)
            loss = loss + lambda_reg * reg
        loss_ce_total += loss.item(); n += 1
        pred = int(torch.argmax(logits, dim=-1).item())
        y_true.append(int(y.item())); y_pred.append(pred)

        if dump_path is not None:
            rows.append({
                "claim": it["claim"],
                "evidence": it["evidence"],
                "gold": ["SUPPORTS","REFUTES","NEI"][y.item()],
                "pred": ["SUPPORTS","REFUTES","NEI"][pred],
                "S": float(S.item()), "R": float(R.item()), "M": float(M.item()),
                "O": float(O.squeeze(0).item())
            })

    from sklearn.metrics import classification_report, accuracy_score, f1_score
    report = classification_report(y_true, y_pred, labels=[0,1,2],
                                   target_names=["SUPPORTS","REFUTES","NEI"],
                                   zero_division=0, output_dict=False)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")

    if dump_path is not None and rows:
        pd.DataFrame(rows).to_csv(dump_path, index=False, encoding="utf-8")

    return {"acc": acc, "macro_f1": f1m, "loss": loss_ce_total/max(n,1), "report": report}

def main():
    ap = argparse.ArgumentParser()
    # ==== 关键：所有默认路径均以 BASE_DIR 为锚 ====
    ap.add_argument("--train_csv", type=str, default=os.path.join(BASE_DIR, "data", "train.csv"))
    ap.add_argument("--val_csv",   type=str, default=os.path.join(BASE_DIR, "data", "val.csv"))
    ap.add_argument("--exp_train", type=str, default=os.path.join(BASE_DIR, "data", "expectations_train.jsonl"))
    ap.add_argument("--exp_val",   type=str, default=os.path.join(BASE_DIR, "data", "expectations_val.jsonl"))
    ap.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--finetune_encoder", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3, help="head-only 1e-3；若微调 encoder 建议 2e-5")
    ap.add_argument("--margin_m", type=float, default=0.0)
    ap.add_argument("--lambda_reg", type=float, default=0.0)
    ap.add_argument("--output_dir", type=str, default=os.path.join(BASE_DIR, "outputs", "eep_supervised"))
    ap.add_argument("--seed", type=int, default=42)

    # 聚合与期望采样
    ap.add_argument("--pooling", type=str, choices=["max","lse","topk"], default="max")
    ap.add_argument("--beta", type=float, default=10.0, help="lse 温度（越大越接近 max）")
    ap.add_argument("--topk", type=int, default=3, help="topk 聚合的 k")
    ap.add_argument("--k_pos", type=int, default=None, help="每条样本保留的正向期望数（None=不限）")
    ap.add_argument("--k_neg", type=int, default=None, help="每条样本保留的反向期望数（None=不限）")

    # 类权重与 γ 冻结
    ap.add_argument("--class_weights", type=str, default=None, help="如 '1,1,1' 对应 S/R/NEI")
    ap.add_argument("--init_gamma", type=float, default=1.0)
    ap.add_argument("--freeze_gamma", action="store_true")

    # 导出验证集预测明细
    ap.add_argument("--dump_val_preds", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    train_items = build_dataset(args.train_csv, args.exp_train)
    val_items   = build_dataset(args.val_csv,   args.exp_val)

    encoder = DeltaEncoder(args.model_name, finetune=args.finetune_encoder).to(device)
    head = EEPHead(init_gamma=args.init_gamma, freeze_gamma=args.freeze_gamma).to(device)

    # 参数与优化器
    params = list(head.parameters())
    if args.finetune_encoder:
        params += list(encoder.parameters())
        print("[INFO] Finetuning encoder with lr", args.lr)
    else:
        print("[INFO] Training only EEP head (scale/bias/gamma/temp), frozen encoder.")

    optimizer = optim.AdamW(params, lr=args.lr)

    # 交叉熵（可选类权重）
    if args.class_weights:
        try:
            w = [float(x) for x in args.class_weights.split(",")]
            assert len(w) == 3
            weight = torch.tensor(w, device=device, dtype=torch.float32)
            ce = nn.CrossEntropyLoss(weight=weight)
            print(f"[INFO] Using class weights: {w}")
        except Exception:
            ce = nn.CrossEntropyLoss()
            print("[WARN] Bad --class_weights, fallback to uniform.")
    else:
        ce = nn.CrossEntropyLoss()

    best_f1 = -1.0
    for ep in range(1, args.epochs+1):
        encoder.train(args.finetune_encoder); head.train()
        total, n = 0.0, 0
        rng = random.Random(args.seed + ep)  # 每轮不同的期望采样
        pbar = tqdm(train_items, desc=f"Epoch {ep}")
        for it in pbar:
            optimizer.zero_grad()
            ev_sents = simple_sent_split(it["evidence"])
            S, R, M = compute_SR_M(
                encoder, it["pos_exp"], it["neg_exp"], ev_sents,
                pooling=args.pooling, beta=args.beta, topk=args.topk,
                k_pos=args.k_pos, k_neg=args.k_neg, rng=rng
            )
            logits, _ = head(S.unsqueeze(0), R.unsqueeze(0), M.unsqueeze(0))
            y = torch.tensor([label_to_idx(it["label"])], device=device)
            loss = ce(logits, y)
            if args.margin_m>0 and args.lambda_reg>0:
                reg = torch.clamp(args.margin_m - torch.abs(S - R), min=0.0)
                loss = loss + args.lambda_reg * reg
            loss.backward()
            optimizer.step()
            total += loss.item(); n += 1
            if n % 50 == 0:
                pbar.set_postfix(loss=f"{total/max(n,1):.3f}")

        tr_loss = total / max(n,1)

        # eval
        dump_path = os.path.join(args.output_dir, "val_preds.csv") if args.dump_val_preds else None
        metrics = evaluate(
            encoder, head, val_items,
            pooling=args.pooling, beta=args.beta, topk=args.topk,
            margin_m=args.margin_m, lambda_reg=args.lambda_reg,
            k_pos=args.k_pos, k_neg=args.k_neg, seed=args.seed+123,
            dump_path=dump_path
        )
        print(f"\n[Epoch {ep}] train_loss={tr_loss:.4f} | val_loss={metrics['loss']:.4f} "
              f"| acc={metrics['acc']:.4f} | macro_f1={metrics['macro_f1']:.4f}")
        print(metrics["report"])

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save({
                "head": head.state_dict(),
                "gamma": float(head.gamma.item())
            }, os.path.join(args.output_dir, "best.pt"))
            with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump({"best_macro_f1": best_f1,
                           "acc": metrics["acc"],
                           "pooling": args.pooling,
                           "beta": args.beta,
                           "topk": args.topk}, f, ensure_ascii=False, indent=2)
            print(f"[OK] Saved best to {args.output_dir}")

if __name__ == "__main__":
    main()
