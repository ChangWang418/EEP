# EEP-CoT-SelfAsk: A Comparative Framework for Structured Factual Reasoning

This project implements and compares three prompting strategies — **EEP**, **CoT**, and **Self-Ask** — across two fact verification datasets: **FEVER** and **PubHealth**.

## 📁 Project Structure

```plaintext
EEP-COT-SELFASK/
├── FEVER/
│   ├── CoT/
│   ├── EEP/
│   ├── Selfask/
│   └── Fever_Data/
│       ├── shared_task_dev.jsonl
│       ├── train.jsonl(Optional).md
│       └── fever.db.md
├── Pubhealth/
│   ├── CoT/
│   ├── EEP/
│   ├── Selfask/
│   └── Pubhealth_Data/
│       ├── dev.tsv
│       ├── test.tsv
│       └── train.tsv(Optional).md
├── .env          ← Put your OpenAI API key here
```

## ✅ How to Run This Project

### 1. Download All Files

Download all files and folders from the project root, including two dataset folders (`FEVER/`, `Pubhealth/`) and the `.env` config file.

### 2. Manually Download External Data

Open the following `.md` files, copy the embedded Google Drive links, download the data, and place them into the paths as instructed:

| File Path                                      | Required | Description                        |
|------------------------------------------------|----------|------------------------------------|
| `FEVER/Fever_Data/fever.db.md`                 | ✅ Yes   | FEVER database (SQLite)            |
| `FEVER/Fever_Data/train.jsonl(Optional).md`    | ⭕ Optional | FEVER training set (JSONL)        |
| `Pubhealth/Pubhealth_Data/train.tsv(Optional).md` | ⭕ Optional | PubHealth training set (TSV)     |

### 3. Add Your OpenAI API Key

Open `.env` and paste your OpenAI API key like this:

```env
OPENAI_API_KEY=sk-xxxxx...your-key...
```

### 4. Run Target Model Script

Each model includes one or more main scripts named with `run_experiment...py`.

Example paths:

```plaintext
FEVER/EEP/run_experiment - db-Random 1998.py
FEVER/CoT/run_experiment - db-result save.py
Pubhealth/Selfask/run_experiment - db-result save.py
...
```

Choose the appropriate model and dataset to run your experiment.

### 5. Handle OpenAI Rate Limits

- OpenAI API may enforce rate limits. It is **strongly recommended** to split execution into time-separated batches.
- All scripts support **auto-resume**, checking for existing result files before proceeding.
- You can modify `batch_size` (default: 500) in the scripts for fine-grained control.

### 6. Combine Results

After running, use the following script to merge partial CSV files into a single result file:

```bash
python combine_csv.py
```

Example paths:

```plaintext
FEVER/EEP/combine_csv.py
Pubhealth/CoT/combine_csv.py
```

The merged output will be saved as `combine.csv`.

### 7. Analyze Statistics

Use `static.py` to analyze and visualize the results in `combine.csv`:

```bash
python static.py
```

This will output:

- Precision, recall, F1-score per class
- F1-score visualization chart

## 📦 Environment Setup

Make sure you are using Python 3.8+ and install the following dependencies:

```bash
pip install openai scikit-learn matplotlib python-dotenv
```

## 📎 FEVER Database Link

You can download the `fever.db` file (required for FEVER experiments) here:

🔗 [fever.db on Google Drive](https://drive.google.com/file/d/1mvkpBHA-8_1EIQD3j3DOWK9EWoGcDKP5/view?usp=drive_link)

---

Feel free to extend the models or adapt this framework to other factual reasoning tasks. For any further questions, please reach out to the project author.
