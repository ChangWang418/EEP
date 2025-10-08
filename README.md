# 🧩 EEP–CoT–SelfAsk–Standard–Decompose: A Comparative Framework for Fact-Checking Prompts

This project implements and compares multiple prompting strategies (**EEP**, **CoT**, **Self-Ask**, **Standard**, **Decompose**) across three fact-checking datasets (**FEVER**, **PubHealth**, **SciFact**).  

The goal is to evaluate how different structured reasoning prompts perform in factual verification tasks with reproducible experiment and analysis workflows.  

---

## 📁 Project Structure

```plaintext
EEP-CoT-SelfAsk-Standard-DECOMP/
├── FEVER/
│   ├── CoT/
│   ├── EEP/
│   ├── Selfask/
│   ├── Standard/
│   ├── Decompose/
│   ├── Fever_Data/
│   │   ├── shared_task_dev.jsonl
│   │   ├── train.jsonl (optional)
│   │   └── fever.db
│   └── Supervised/                    ← Optional supervised EEP extension
│       ├── data/                      ← Stores intermediate supervised data (train/val CSVs & expectation JSONL)
│       ├── prepare_fever_subset.py
│       ├── make_expectations.py
│       └── train_eep_delta.py
├── Pubhealth/
│   ├── CoT/
│   ├── EEP/
│   ├── Selfask/
│   ├── Standard/
│   ├── Decompose/
│   └── Pubhealth_Data/
│       ├── dev.tsv
│       ├── test.tsv
│       └── train.tsv (optional)
├── SCI/
│   ├── CoT/
│   ├── EEP/
│   ├── Selfask/
│   ├── Standard/
│   ├── Decompose/
│   └── Sci_Data/
│       ├── claims_dev.jsonl
│       └── corpus.jsonl
├── .env     ← Place your OpenAI API key here
```

---

## ✅ Usage

### 1️⃣ Download the Repository
Clone or copy the full project structure, including the three dataset folders (`FEVER/`, `Pubhealth/`, `SCI/`) and the `.env` configuration file.  

---

### 2️⃣ Dataset Preparation

Ensure that data files such as `FEVER/Fever_Data/fever.db` exist.  
If missing, refer to the official dataset page or download from the link below.  

> 🔗 **FEVER Database Download Link (Google Drive)**  
> [Download here](https://drive.google.com/file/d/1mvkpBHA-8_1EIQD3j3DOWK9EWoGcDKP5/view?usp=drive_link)

---

### 3️⃣ Configure API Key
Create or edit the `.env` file in the project root:  
```env
OPENAI_API_KEY=sk-xxxxx...your-key...
```

---

### 4️⃣ Run Prompt Experiments (Few-shot / Full-sample)

Each prompting strategy has its own entry script, for example:  
```bash
python FEVER/CoT/run_experiment_db-result-save.py
python Pubhealth/EEP/run_experiment_db-Random1998.py
python SCI/Decompose/run_experiment_db-result-save.py
```

These scripts reproduce **few-shot** or **full-sample** factual verification experiments.  
You can adjust the number of examples or prompt design to compare model performance under different conditions.  

---

### 5️⃣ Combine Experiment Results
For experiments run in multiple batches, use `combine_csv.py` to merge results:  
```bash
python combine_csv.py
```
Example locations:
```
FEVER/CoT/combine_csv.py
Pubhealth/Standard/combine_csv.py
SCI/Selfask/combine_csv.py
```

---

### 6️⃣ Evaluation and Visualization
Run `static.py` to generate evaluation metrics:  
```bash
python static.py
```

Outputs include:  
- Per-class Precision / Recall / F1  
- Macro-F1 bar chart  

---

## 📦 Environment

Recommended: Python 3.8+  
Install dependencies:  
```bash
pip install openai scikit-learn matplotlib torch transformers pandas tqdm python-dotenv
```

---

## 📌 Framework Summary

| Category | Description |
|-----------|--------------|
| **Datasets** | FEVER + PubHealth + SciFact |
| **Prompting Strategies** | EEP, CoT, Self-Ask, Standard, Decompose |
| **Reproducibility** | Unified relative paths and `.env` configuration |
| **Outputs** | Structured logs, JSON metrics, resumable execution |

You can freely combine datasets and prompting strategies to compare model performance across methods.  

---

## 🧠 Supervised EEP Extension

This module provides a lightweight supervised fine-tuning pipeline to improve factual stability while keeping EEP interpretable.  

---

### 📂 Workflow

1️⃣ **Prepare the dataset**  
```bash
python FEVER/Supervised/prepare_fever_subset.py
```
Reads `Fever_Data/train.jsonl` and `fever.db`  
→ Outputs: `Supervised/data/train.csv` and `val.csv`

2️⃣ **Build expectations**  
```bash
python FEVER/Supervised/make_expectations.py
```
Generates:
```
Supervised/data/expectations_train.jsonl
Supervised/data/expectations_val.jsonl
```

3️⃣ **Train the supervised EEP model**  
```bash
python FEVER/Supervised/train_eep_delta.py
```
Outputs stored in:
```
Supervised/outputs/eep_supervised/
```
Includes `best.pt`, `metrics.json`, and optional `val_preds.csv`.

---

### 🎯 Highlights
- Balanced FEVER subset (1k per class by default)  
- Hybrid expectation generation (positive + negated clauses)  
- Lightweight supervision avoiding overfitting  
- Directly comparable with few-shot / full-sample prompt results  

---

### 🧩 Example Training Commands

Below are recommended training commands for different pooling modes and encoder configurations (run inside `FEVER/Supervised`):  

#### **1️⃣ Baseline (Max Pooling, γ = 1.0)**
```bash
python train_eep_delta.py --train_csv data/train.csv --val_csv data/val.csv --exp_train data/expectations_train.jsonl --exp_val data/expectations_val.jsonl --epochs 5 --lr 1e-3 --pooling max --k_pos 3 --k_neg 3 --seed 42 --dump_val_preds
```

#### **2️⃣ Soft Pooling (LSE, γ = 0.6, lower NEI weight)**
```bash
python train_eep_delta.py --train_csv data/train.csv --val_csv data/val.csv --exp_train data/expectations_train.jsonl --exp_val data/expectations_val.jsonl --epochs 5 --lr 1e-3 --pooling lse --beta 15 --init_gamma 0.6 --class_weights "1.2,1.2,0.8" --k_pos 2 --k_neg 2 --seed 42 --dump_val_preds
```

#### **3️⃣ Top-k Pooling (more robust)**
```bash
python train_eep_delta.py --train_csv data/train.csv --val_csv data/val.csv --exp_train data/expectations_train.jsonl --exp_val data/expectations_val.jsonl --epochs 5 --lr 1e-3 --pooling topk --topk 3 --init_gamma 0.8 --class_weights "1.1,1.1,0.9" --k_pos 2 --k_neg 2 --seed 42 --dump_val_preds
```

#### **4️⃣ Stronger Encoder (MPNet-base)**
```bash
python train_eep_delta.py --train_csv data/train.csv --val_csv data/val.csv --exp_train data/expectations_train.jsonl --exp_val data/expectations_val.jsonl --epochs 5 --lr 1e-3 --pooling lse --beta 20 --init_gamma 0.6 --class_weights "1.2,1.2,0.8" --k_pos 2 --k_neg 2 --seed 42 --dump_val_preds --model_name sentence-transformers/all-mpnet-base-v2
```

---

📘 **Notes**
- All model checkpoints are saved in `Supervised/outputs/eep_supervised/`.  
- You can view and compare results directly in `metrics.json`.  
- Adjust `gamma` or `class_weights` to balance stability and bias.  

---

### 🔬 Summary

The **Supervised EEP** module bridges **structured prompting** and **parameter-efficient learning**.  
It preserves interpretability while improving factual consistency through limited supervision.  
