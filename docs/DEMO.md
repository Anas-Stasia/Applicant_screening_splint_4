# Final demo & technical presentation (Sprint 4)

A run of show for a ~15-minute slot: about 7 minutes of slides, then an 8-minute live demo.
Everything in the demo runs from this repo.

## Before you present (T-30 min)

```bash
docker compose up --build -d                        # or use the deployed URL
python serving/scripts/smoke_test.py --url http://localhost:8000 --expect-model all-MiniLM-L6-v2
```
Open these tabs: `/demo`, `/docs`, `/v1/monitoring/drift`, `/metrics`, the GitHub Actions run of the
last merged PR, and `data_pipeline/sprint4_production.ipynb` (already run, as a fallback if the network fails).

## Presentation outline (slides)

| # | Slide | Key message | Evidence |
|---|---|---|---|
| 1 | Problem | Recruiters can't read every resume, so a first-pass screener ranks and flags. A person decides | Sprint 1 framing |
| 2 | Data & pipeline | 2,484 → 2,481 clean Kaggle resumes, 24 categories, a reproducible notebook pipeline | Sprint 1 numbers |
| 3 | Model choice | MiniLM embeddings + cosine. Zero-shot over fine-tuning (no labels) | 66.1% top-1 / 81.7% top-3, vs TF-IDF 63.4% / 84.9% |
| 4 | Guardrails | `ok` / `low_confidence` / `rejected`, and a 0.35 floor | The saxophonist case |
| 5 | Architecture | Client → FastAPI → MatchingService → cache → encoder. Every prediction → log → drift → metrics | `serving/app/main.py` diagram |
| 6 | Production | One image runs on Render, Railway and AWS. `/ready` gate. Smoke test after every deploy | `docs/DEPLOYMENT.md` |
| 7 | Monitoring | No labels in production, so watch inputs and confidence. PSI with noise-aware thresholds | False-alarm table in `docs/MONITORING.md` |
| 8 | Team workflow | GitHub flow: PR → CI (lint, tests, Docker smoke) → review → squash-merge → deploy | A green Actions run |
| 9 | What tests caught | A notebook with Windows paths, and training/serving skew in text cleaning | `test_pipeline.py` |
| 10 | Limits & next | 256-token truncation, untuned floor, no outcome labels yet, per-replica cache | Next steps |

## Live demo script (8 minutes)

**1. The product (2 min): `/demo`**
Pick the category ACCOUNTANT. Paste two real resumes and one blank one. Show the ranking, the
`low_confidence` badge, and the rejected item with its reason. *"A blank upload doesn't break
the batch. It comes back with a reason the UI can show."*

**2. The contract (1 min): `/docs`**
Open `POST /v1/screen` and send a payload with a typo (`"txt"`). Show the 422 with the field name.
*"Both squads build against this spec. Typos fail loudly."*

**3. What an operator sees (2 min)**
`GET /v1/monitoring/predictions?limit=5`: the requests just made, with request ids and
statuses, *and no resume text*. Then `/metrics`: request counts and latency histogram.

**4. Drift (2 min)**
Run the "campaign" cell from `sprint4_production.ipynb` §2b against the live server, or this:
```bash
python - <<'PY'
import sys; sys.path.insert(0, "serving/clients")
import pandas as pd
from screening_client import ScreeningClient
df = pd.read_csv("data_pipeline/data/processed/resumes_clean.csv")
shift = df[df.Category.isin(["ARTS", "APPAREL", "AUTOMOBILE"])].sample(200, random_state=2)
api = ScreeningClient("http://localhost:8000")
api.match_batch([{"id": str(i), "text": t} for i, t in zip(shift.ID, shift.clean_text)])
api.match_batch([{"id": f"blank{i}", "text": ""} for i in range(30)])
PY
```
Refresh `/v1/monitoring/drift`: `category_psi` and `similarity_psi` are on **alert** and
`rejection_rate` fires. Walk through one row of the runbook in `docs/MONITORING.md`.

**5. Ship a change (1 min)**
Show a PR: the CI checks (lint, test 3.11/3.12, docker), the quality gate test, and the merge.
Then show Render deploying only after the checks passed (`checksPass`).

## Likely questions

* **"How accurate is it?"** 66.1% top-1 and 81.7% top-3 on the cohort. The job descriptions were built
  from the resumes' own vocabulary, though, so this is a sanity check and not a hiring-outcome measure.
* **"Is it fair?"** Fairness auditing was out of scope (Sprint 1). The tool ranks and flags, and a
  person decides. Logging `request_id` lets outcomes be joined to predictions later for an audit.
* **"Why not fine-tune?"** There are no labelled hiring decisions, and zero-shot needs no retraining for
  new categories (Sprint 2).
* **"Does it scale?"** It's stateless apart from the cache, so add containers. Each holds its own model
  (~400 MB). A shared Redis cache and request micro-batching are the next steps.
* **"What if the drift alert fires?"** See the runbook. Most alerts trace to an upstream client
  (empty or truncated text) or to a real change in who is applying.
