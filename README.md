# 🧩 EEP–CoT–SelfAsk–Standard–Decompose: A Comparative Framework for Fact-Checking Prompts

This project implements and compares multiple prompting strategies (**EEP**, **CoT**, **Self-Ask**, **Standard**, **Decompose**) across three fact-checking datasets (**FEVER**, **PubHealth**, **SciFact**).  

The goal is to evaluate how different structured reasoning prompts perform in factual verification tasks, with reproducible experiment and analysis workflows.  

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
│   └── Fever_Data/
│       ├── shared_task_dev.jsonl
│       ├── train.jsonl(Optional).md
│       └── fever.db.md
├── Pubhealth/
│   ├── CoT/
│   ├── EEP/
│   ├── Selfask/
│   ├── Standard/
│   ├── Decompose/
│   └── Pubhealth_Data/
│       ├── dev.tsv
│       ├── test.tsv
│       └── train.tsv(Optional).md
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

### 1. Download the repository
Clone or copy the full project structure, including three dataset folders (`FEVER/`, `Pubhealth/`, `SCI/`) and the `.env` configuration file.  

### 2. Manually download external datasets
Open the `.md` files in each dataset folder and download the linked Google Drive files. Place them in the specified directory.  

- `FEVER/Fever_Data/fever.db.md` → FEVER database (required)  
- `FEVER/Fever_Data/train.jsonl(Optional).md` → FEVER training set (optional)  
- `Pubhealth/Pubhealth_Data/train.tsv(Optional).md` → PubHealth training set (optional)  

### 3. Configure API Key
Create or edit `.env` in the project root:  
```env
OPENAI_API_KEY=sk-xxxxx...your-key...
```

### 4. Run experiment scripts
Each model directory contains a `run_experiment...py` file as the entry point.  

Examples:
```plaintext
FEVER/CoT/run_experiment - db-result save.py
Pubhealth/EEP/run_experiment - db-Random 1998.py
SCI/Decompose/run_experiment - db-result save.py
```

Choose the dataset and prompting strategy you want to run.  

### 5. Automatic checkpointing
All scripts support **resumable execution**. If interrupted, they continue from the last CSV file.  
You can adjust `BATCH_SIZE` (default 500) to control per-batch size.  

### 6. Combine results
Use `combine_csv.py` to merge partial results:  
```bash
python combine_csv.py
```
Example locations:
```plaintext
FEVER/CoT/combine_csv.py
Pubhealth/Standard/combine_csv.py
SCI/Selfask/combine_csv.py
```

The merged file is saved as `combine.csv`.  

### 7. Statistics & Visualization
Run `static.py` to generate evaluation metrics:  
```bash
python static.py
```

Outputs include:
- Precision, Recall, F1 for each class  
- F1-score bar chart  

---

## 📦 Environment

Recommended: Python 3.8+  
Install dependencies:  
```bash
pip install openai scikit-learn matplotlib python-dotenv
```

---

## 🔑 Database Download Links

FEVER requires an external `fever.db`:  
🔗 [fever.db on Google Drive](https://drive.google.com/file/d/1mvkpBHA-8_1EIQD3j3DOWK9EWoGcDKP5/view?usp=drive_link)  

---

## 📌 Summary

Compared to the initial version, this framework expands in two major ways:  
- **Datasets**: from FEVER + PubHealth → now includes SciFact  
- **Prompting Strategies**: from EEP, CoT, Self-Ask → now includes Standard and Decompose  
- **Reproducibility**: unified relative paths and `.env` config for easier reuse and extension  

You can run any dataset–strategy combination and directly compare performance across methods.  
