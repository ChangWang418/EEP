# EEP-Sup Minimal Trial (T5 + LoRA on a tiny FEVER subset)

This is a **lowest-cost feasibility check** to see if a small supervised model benefits your EEP framing.

## What you get
- `prepare_fever_subset.py`: grabs a small FEVER split (500 rows) and builds a toy CSV with columns: `claim,evidence,label`.
- `train_t5_lora.py`: LoRA-fine-tunes `t5-base` as a *generative classifier* (claim+evidence -> label).
- `evaluate.py`: computes accuracy, macro-F1, classification report on the held-out set.

> If this quick run underperforms vs. your GPT results, you can legitimately report that a supervised small model did not beat zero-shot GPT, and skip further investment.

## Quickstart

1) Create a fresh environment and install deps:
```bash
pip install -U pip
pip install -U datasets transformers peft accelerate evaluate scikit-learn
```

2) Prepare a mini FEVER subset (random 500 examples). This script will try HuggingFace `fever` first.
```bash
python prepare_fever_subset.py --seed 42 --n_samples 500
```

This writes:
- `data/train.csv` (400 rows)
- `data/val.csv`   (100 rows)

3) Train T5-base with LoRA (1 epoch). Tweak batch size if OOM.
```bash
python train_t5_lora.py   --train_path data/train.csv   --val_path data/val.csv   --output_dir outputs/t5_lora_trial   --epochs 1   --batch_size 8   --lr 5e-5
```

4) Evaluate
```bash
python evaluate.py   --val_path data/val.csv   --model_dir outputs/t5_lora_trial
```

You will get overall accuracy and macro-F1. If macro-F1 is clearly below your GPT baseline (e.g., in your paper FEVER macro-F1 ~ 86 with EEP), you can stop here and document as a negative finding. If performance looks promising, keep as "Proof-of-Concept" (Appendix).

## Notes
- This is a plain supervised baseline (claim+evidence->label). To mimic EEP-Sup more closely, you can swap the `build_input()` function to include brief "expectation" scaffolding. Keep it short to remain a fast trial.
- You can increase `--n_samples` later to 1k/2k if you want a slightly more stable signal.
- Works on CPU (slow) or single GPU; LoRA keeps memory modest.
