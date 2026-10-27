# Applicant Screening & Recommendation Tool
### Sprint 1: Product & Data Pipeline · Sprint 2: Modeling & Evaluation · Sprint 3: Serving & Integration · Sprint 4: Team & Production

[![ci](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

A first-pass resume screener: it ranks applicants against a job with MiniLM sentence
embeddings and flags the ones it isn't sure about. A person makes the hiring decision.

## Quick start

```bash
docker compose up --build          # production image → http://localhost:8000/docs  ·  /demo
```
or without Docker:
```bash
cd serving
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
pip install -r requirements-dev.txt && pytest -q     # no model download needed
```
A fresh clone works without running any notebook: the server loads the committed bundle in
`serving/artifacts/`. Once you've run the notebooks, it uses `data_pipeline/data/processed/` instead.

## Sprint 4 — Team & Production (new)

| Checklist item | Where |
|---|---|
| Docker containerization for ML microservices | `serving/Dockerfile`, `docker-compose.yml`, `.dockerignore`, `serving/scripts/export_artifacts.py` |
| Cloud deployment (Render / AWS / Railway) | `render.yaml`, `railway.json`, `deploy/aws/deploy_ecs_express.sh`, [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) |
| Prediction logging & monitoring data drift | `serving/app/monitoring.py`, `GET /v1/monitoring/drift`, `GET /metrics`, `serving/scripts/drift_report.py`, [`docs/MONITORING.md`](docs/MONITORING.md) |
| GitHub flow & automated testing for ML pipelines | `.github/workflows/ci.yml`, `serving/tests/test_pipeline.py`, [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) |
| Final demo and technical presentation | [`docs/DEMO.md`](docs/DEMO.md), `data_pipeline/sprint4_production.ipynb` |

* **Logging:** one JSON line per prediction (request id, model, build, status, score, latency, text length
  and hash). The resume text itself is never logged.
* **Drift:** recent traffic is compared with the training cohort on six checks (PSI on similarity,
  category mix and text length, the low-confidence and rejection rates, and the embedding centroid).
  The thresholds account for small-window noise.
* **Deploy:** one image (model baked in, non-root, `$PORT`, `/ready` health check) for Render, Railway or
  AWS ECS Express Mode, with `smoke_test.py` after every deploy.
* **CI:** every PR runs ruff, pytest on Python 3.11 and 3.12, and a Docker build with a smoke test of the
  real model. Pipeline tests include a **top-1 ≥ 62% quality gate** and a training/serving text-parity check.
* **Fixed along the way:** `tfidf_baseline.ipynb` (still at the repo top level, with `C:\Users\…` paths)
  moved into `data_pipeline/`. The API's HTML cleaning now matches `preprocess.ipynb` exactly
  (13 of 4,968 raw texts differed before). `main.ipynb` now runs all five Sprint 1 stages.

## Sprint 3 — Serving & Integration
The model runs behind a FastAPI server with batch inference, in [`serving/`](serving/README.md):

* `serving/README.md` covers architecture, endpoints, error codes, configuration, monitoring, and how the
  Frontend and Backend squads integrate.
* `data_pipeline/sprint3_serving.ipynb` works through each Sprint 3 checklist item against the live app.
* `serving/clients/` has ready-made JavaScript (frontend) and Python (backend) clients.
  `serving/demo/` is a reference UI served at `/demo`.

---


## What this does
Builds a full pipeline that screens/ranks applicant resumes against job descriptions:
1. **config.ipynb** — paths & constants (run first, or via `%run` in the other notebooks)
2. **download_data.ipynb** — pulls the [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
3. **preprocess.ipynb** — cleans resume text, derives one job description per category
4. **tfidf_baseline.ipynb** — tokenization, stop words, TF-IDF matching baseline
5. **embeddings.ipynb** — Sentence-Transformers semantic matching (upgrades the TF-IDF baseline)
6. **main.ipynb** — runs steps 2–5 in order, end-to-end
7. **main_sprint2.ipynb** — Sprint 2 evaluation (similarity theory, zero-shot, accuracy, edge cases)
8. **sprint3_serving.ipynb** / **sprint4_production.ipynb** — serving and production walkthroughs

After re-running the pipeline, refresh what the server ships with:
`python serving/scripts/export_artifacts.py`

## Setup
```bash
pip install -r requirements.txt
```

Get a Kaggle API key (one-time):
1. kaggle.com → Account Settings/Profile → API → Create New Token
2. Save it to `~/.kaggle/access_token` (or `~/.kaggle/kaggle.json`, depending on your
   Kaggle client version) — see `download_data.ipynb` for exact commands
3. On Mac/Linux: `chmod 600` the token file

## Run
Open `main.ipynb` in VS Code (with the Jupyter extension enabled) and click **Run All**.
This runs every stage in order. Each notebook can also be opened and run individually —
each one's first cell is `%run config.ipynb`, so it's self-contained.

Alternative: if the Kaggle API gives you trouble, download the dataset manually from the
link above and place `Resume.csv` at `data_pipeline/data/raw/Resume.csv`, then skip
straight to `preprocess.ipynb`.

## Project structure
```
applicant_screening_nb/
├── README.md  requirements.txt  .gitignore  .dockerignore
├── docker-compose.yml         # Sprint 4: local production-like run
├── render.yaml  railway.json  # Sprint 4: cloud deploy configs
├── deploy/aws/                # Sprint 4: ECS Express Mode deploy script
├── .github/
│   ├── workflows/ci.yml       # Sprint 4: lint → tests → Docker smoke → publish → deploy
│   ├── pull_request_template.md
│   └── dependabot.yml
├── docs/                      # Sprint 4: DEPLOYMENT, MONITORING, CONTRIBUTING, DEMO
├── serving/                   # Sprint 3–4: FastAPI service (see serving/README.md)
│   ├── app/                   #   main.py, schemas.py, service.py, monitoring.py, ...
│   ├── artifacts/             #   committed bundle the server loads (+ drift reference, manifest)
│   ├── clients/  demo/  tests/  scripts/
│   ├── Dockerfile
│   └── requirements*.txt
└── data_pipeline/
    ├── config.ipynb  download_data.ipynb  preprocess.ipynb
    ├── tfidf_baseline.ipynb  embeddings.ipynb  main.ipynb           # Sprint 1
    ├── similarity_theory.ipynb  huggingface_transformers.ipynb       # Sprint 2
    ├── classification_approach.ipynb  evaluate_recommendations.ipynb
    ├── edge_cases.ipynb  main_sprint2.ipynb
    ├── sprint3_serving.ipynb                                         # Sprint 3
    └── sprint4_production.ipynb                                      # Sprint 4
data_pipeline/data/            # created on first run — gitignored, not committed
├── raw/
└── processed/
    ├── tfidf/                 # saved TF-IDF vectorizer + matrices
    └── embeddings/            # saved Sentence-Transformer embeddings
```

## Version control setup
Day-to-day work follows GitHub flow. Branches, PRs, required checks and the rules for
model/data changes are in [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md). The original
one-time setup is below.

Run these once, from the `applicant_screening_nb/` folder:
```bash
git init
git add .
git commit -m "Sprint 1: product framing, data pipeline, TF-IDF baseline, embeddings"
```
`.gitignore` is already set up to exclude the `data/` folder (regenerated by the
pipeline, not source; the slim `serving/artifacts/` bundle is the one exception, so CI and
cloud builds work from a clone), credentials (`kaggle.json`, `access_token`), and standard
Python/Jupyter clutter — so only pipeline code gets tracked, keeping the repo
reproducible: anyone can clone it and regenerate the same results by running the
notebooks, rather than relying on committed data files.

To connect it to GitHub (optional, once you've created an empty repo there):
```bash
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

## Design notes
- The Kaggle dataset ships resumes with a `Category` label but no separate job postings,
  so `preprocess.ipynb` derives a pseudo job description per category from that
  category's most frequent resume vocabulary.
- `tfidf_baseline.ipynb` is intentionally a *baseline* — it only catches exact word
  overlap, so it misses semantic matches (e.g. "developer" vs "engineer").
  `embeddings.ipynb` closes that gap using pretrained Sentence-Transformers embeddings
  and includes a side-by-side comparison against the TF-IDF results.
