# EEP-CoT-SelfAsk Fact Verification

This project evaluates different prompt strategies (EEP, CoT, and Self-Ask) for factual consistency in large language models, based on FEVER and PubHealth datasets.

## 📦 How to Use

1. **Download** all files and folders in the root directory (2 folders and one environment config file `.env`).
2. **Manually open** the following markdown files and download the Google Drive content inside. Place the downloaded files as instructed:
   - `FEVER/Fever_Data/Fever.db.md` (**Required**)
   - `FEVER/Fever_Data/train.jsonl(Optional).md` *(Optional)*
   - `Pubhealth/Pubhealth_Data/train.tsv(Optional).md` *(Optional)*
3. **Insert your OpenAI API key** into the `.env` file.
4. **Run** your target experiment script (files beginning with `run_experiment...`).
5. **Note**: OpenAI APIs may be rate-limited. It is recommended to run the scripts in intervals. The program automatically resumes from saved batches (default `batch_size = 500`, adjustable in the script).
6. **Use** `combine.py` to merge output files and generate `combine.csv`.
7. **Use** `static.py` to analyze `combine.csv` and produce visual reports.

## 📁 Structure

```
EEP-CoT-SelfAsk/
├── FEVER/
│   └── Fever_Data/
│       ├── Fever.db.md
│       └── train.jsonl(Optional).md
├── Pubhealth/
│   └── Pubhealth_Data/
│       └── train.tsv(Optional).md
├── .env
├── run_experiment_*.py
├── combine.py
└── static.py
```

## 🧠 Notes

- Ensure all dependencies are installed.
- Keep all files relative to the root for correct path resolution.