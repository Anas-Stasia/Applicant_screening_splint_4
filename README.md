# Applicant Screening & Recommendation Tool — Data Pipeline (Sprint 1)

## What this does
Downloads the Kaggle "Resume Dataset" and turns it into two clean, reusable files:
- `data/processed/resumes_clean.csv` — cleaned resume text per applicant
- `data/processed/job_descriptions.csv` — one representative job description per job category

## Setup
```bash
pip install -r requirements.txt
```

Get a Kaggle API key (one-time):
1. kaggle.com → Account Settings → Create New API Token → downloads `kaggle.json`
2. Move it to `~/.kaggle/kaggle.json`
3. `chmod 600 ~/.kaggle/kaggle.json`

## Run
```bash
cd data_pipeline
python main.py
```

This runs both steps: `download_data.py` (collection) then `preprocess.py` (cleaning).
Each can also be run individually.

## Project structure
```
applicant_screening/
├── requirements.txt
├── README.md
└── data_pipeline/
    ├── config.py         # paths, constants — single source of truth
    ├── download_data.py  # Step 1a: pulls raw CSV from Kaggle
    ├── preprocess.py      # Step 1b: cleans text, derives job descriptions
    └── main.py            # orchestrates both steps
data/                      # created on first run — not committed to git
├── raw/
└── processed/
```

## Notes on design choices
- The Kaggle dataset ships resumes with a `Category` label but no separate job postings,
  so `preprocess.py` derives a pseudo job description per category from that category's
  most frequent resume vocabulary. This gives Sprint 1's matching problem two sides
  (resumes + job descriptions) to compare later with TF-IDF and embeddings.
- Cleaning drops resumes under 200 characters (likely parse failures) and de-duplicates
  identical text — both configurable in `config.py`.
- `data/` is not committed to version control (see `.gitignore` once Git setup is added
  in the last checklist item) — only the pipeline code is tracked, so results are
  regenerated from source rather than stored as static files.
