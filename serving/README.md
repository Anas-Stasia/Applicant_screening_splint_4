# Serving, Monitoring & Deployment (Sprints 3–4)

A FastAPI service that wraps the Sprint 1 embedding model (`all-MiniLM-L6-v2`) with
the Sprint 2 guardrails, adds batch inference, and gives the Frontend and Backend
squads a documented, versioned API to build against (Sprint 3).

Sprint 4 made it production-ready: every prediction is logged without the resume text,
drift against the training cohort is checked continuously and exported to Prometheus,
one Docker image deploys to Render, Railway or AWS, and CI tests the ML pipeline on
every pull request. Deeper guides: [`docs/MONITORING.md`](../docs/MONITORING.md),
[`docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md), [`docs/CONTRIBUTING.md`](../docs/CONTRIBUTING.md).

```
serving/
├── app/
│   ├── main.py        # FastAPI app: routes, middleware, startup (lifespan)
│   ├── schemas.py     # Pydantic request/response models = the API contract
│   ├── service.py     # inference: validate → cache → batch encode → cosine → top-k
│   ├── encoders.py    # Sentence-Transformers encoder (+ offline stand-in for tests)
│   ├── errors.py      # one error envelope for every failure
│   ├── monitoring.py  # Sprint 4: prediction log, drift monitor (PSI), Prometheus metrics
│   ├── text.py        # identical cleaning to preprocess.ipynb (BeautifulSoup)
│   └── settings.py    # every knob, overridable by SCREENING_* env vars
├── artifacts/         # Sprint 4: committed ~6 MB bundle the server loads (export_artifacts.py)
├── clients/
│   ├── screening_client.js   # Frontend: browser / Node fetch client
│   └── screening_client.py   # Backend: httpx client with retries + batch chunking
├── demo/index.html    # reference UI, served at /demo
├── tests/             # API, monitoring, ML-pipeline and deployment tests (no model download)
├── scripts/
│   ├── benchmark.py         # latency p50/p95 against a live server
│   ├── export_artifacts.py  # Sprint 4: build/verify artifacts/ from the notebooks' output
│   ├── drift_report.py      # Sprint 4: offline drift report over the JSONL logs
│   └── smoke_test.py        # Sprint 4: post-deploy check (stdlib only)
├── Dockerfile
├── ruff.toml
└── requirements*.txt  # core (no torch) · full (+ sentence-transformers) · dev (tests, lint)
```

## Run it

The server loads `data_pipeline/data/processed/` if you've run the Sprint 1 notebooks,
and otherwise the committed bundle in `serving/artifacts/`, so a fresh clone works
straight away. From this `serving/` folder:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

| URL | What |
|---|---|
| http://localhost:8000/docs | Interactive Swagger docs — try every endpoint in the browser |
| http://localhost:8000/redoc | Read-only reference docs |
| http://localhost:8000/openapi.json | Machine-readable spec (generate client types from it) |
| http://localhost:8000/demo | Small reference UI |

The first start downloads the model (~80 MB). Startup log should say
`loaded job_embeddings.npy` and `loaded resume_embeddings.npy`, meaning Sprint 1's
saved embeddings were reused instead of recomputed.

Tests run in a few seconds. They use a stand-in encoder, synthetic data and the committed
MiniLM embeddings, so nothing is downloaded:

```bash
pip install -r requirements-dev.txt
pytest            # + ruff check .
```

Production-like, from the repo root: `docker compose up --build`.

## Endpoints

| Method | Path | Use it for |
|---|---|---|
| GET | `/health` | Liveness probe. Always 200 if the process is up |
| GET | `/ready` | Readiness probe. 200 once the model is loaded |
| GET | `/v1/categories` | The 24 job categories. Use these to fill dropdowns |
| GET | `/v1/model` | Model name, limits, cache hit/miss stats |
| POST | `/v1/match` | One resume → its best job categories |
| POST | `/v1/match/batch` | Up to 64 resumes in one call. Bad items are reported per item; they don't fail the whole batch |
| POST | `/v1/screen` | **Main screening call:** rank a list of applicants for one job (category *or* free-text description) |
| POST | `/v1/cohort/rank` | Top-N resumes from the 2,481-resume Kaggle cohort for a job |
| GET | `/v1/monitoring/drift` | **Sprint 4.** Recent traffic vs. the training cohort: six checks (PSI, rates, centroid), each `ok` / `warning` / `alert` |
| GET | `/v1/monitoring/predictions` | **Sprint 4.** The last N logged predictions (`?limit=`, `?endpoint=`). No resume text |
| GET | `/metrics` | **Sprint 4.** Prometheus metrics: requests, latency histogram, predictions by status, cache, drift gauges |

### Example: screen applicants for a job

```bash
curl -X POST localhost:8000/v1/screen -H "Content-Type: application/json" -d '{
  "job": {"description": "Payroll accountant: general ledger, month-end close, tax filings"},
  "candidates": [
    {"id": "a-17", "text": "CPA with 6 years in payroll, reconciliations and tax returns..."},
    {"id": "a-18", "text": "Line cook, menu prep, food safety certified..."},
    {"id": "a-19", "text": "hi"}
  ],
  "top_k": 10
}'
```

```json
{
  "job": "custom",
  "ranked": [
    {"id": "a-17", "rank": 1, "similarity": 0.61, "status": "ok", "warnings": []},
    {"id": "a-18", "rank": 2, "similarity": 0.12, "status": "low_confidence", "warnings": ["Similarity 0.120 is below the 0.35 floor."]}
  ],
  "rejected": [
    {"id": "a-19", "rank": null, "similarity": null, "status": "rejected", "warnings": ["TEXT_TOO_SHORT: Text is only 2 characters ..."]}
  ],
  "model": "all-MiniLM-L6-v2",
  "latency_ms": 41.3
}
```
*(The similarity numbers above are only examples.)*

### Result `status`

| status | Meaning | What the UI should do |
|---|---|---|
| `ok` | Confident match | Show the score |
| `low_confidence` | Best similarity is below `SIMILARITY_FLOOR` (0.35), so the resume may be outside the 24 categories (the saxophonist case from `edge_cases.ipynb`) | Show a "needs manual review" badge |
| `rejected` | Empty, too short, no letters, or too long | Show the warning. Nothing was scored |

## Checklist coverage

### 1. FastAPI REST service architecture for ML
- **App factory + lifespan** (`main.create_app`). The model and embeddings load once,
  before the server accepts traffic. They are never loaded per request.
- **Layers.** Routes only handle HTTP. `MatchingService` does the ML work.
  `Encoder` is a small interface, so you can swap in a bigger model
  (`SCREENING_MODEL_NAME=all-mpnet-base-v2`) without touching any routes.
- **Sync `def` endpoints on purpose.** Inference is CPU-bound, so FastAPI runs these
  endpoints in its threadpool and the event loop stays free.
  An `async def` endpoint that calls the model would block every other request.
- **Versioned paths** (`/v1/...`) let us ship `/v2` later without breaking the squads.
- **Ops endpoints** `/health` and `/ready` for Docker/Kubernetes probes.

### 2. Request/response payload schemas (Pydantic)
- `schemas.py` is the contract. FastAPI generates `/docs` and `/openapi.json` from it.
- `extra="forbid"` turns client typos (`"txt"` instead of `"text"`) into a 422.
  Without it, the typo would be silently ignored.
- `str_strip_whitespace`, numeric bounds (`top_k` 1–24), and a cross-field rule
  (`job` takes exactly one of `category` / `description`) are all checked before any model code runs.
- Response models make sure the output always has the documented shape.

### 3. Low-latency inference & model caching
| Technique | Where | Effect |
|---|---|---|
| Load model once at startup + warm-up call | `lifespan`, `service.load` | No cold-start penalty on the first user |
| Reuse Sprint 1 `.npy` embeddings | `_load_or_encode` | Startup skips encoding 2,481 resumes. A one-row spot check confirms the file came from the same model, and it is recomputed if not |
| LRU embedding cache (hash of cleaned text) | `EmbeddingCache` | Re-screening the same applicant needs no forward pass (`"cached": true`) |
| Batch encoding + in-batch de-duplication | `service._embed` | N resumes take 1 model call, not N |
| Normalised vectors | encoders | Cosine similarity is a single matrix product |
| `argpartition` top-k | `rank_cohort` | Selection is O(n), not a full sort of the cohort |
| GZip | middleware | Smaller large batch responses |

Measure on your own machine with the server running:

```bash
python scripts/benchmark.py --url http://localhost:8000 --n 50
```
It reports p50/p95 latency for cold and warm requests, and one batch call vs. the
same resumes sent one by one.

### 4. Error handling & batch prediction endpoints
Every error has the same shape, so client code only needs one error path:

```json
{"error": {"code": "TEXT_TOO_SHORT", "message": "...", "details": null, "request_id": "9f2c1a..."}}
```

| HTTP | `code` | When |
|---|---|---|
| 401 | `UNAUTHORIZED` | `SCREENING_API_KEY` is set and the `X-API-Key` header is missing or wrong |
| 404 | `NOT_FOUND` | Unknown path |
| 413 | `BATCH_TOO_LARGE` | More than `MAX_BATCH_SIZE` items (`details` gives the limit) |
| 422 | `VALIDATION_ERROR` | Bad payload shape (`details` lists each bad field) |
| 422 | `EMPTY_TEXT` / `TEXT_TOO_SHORT` / `NO_TEXT_CONTENT` / `TEXT_TOO_LONG` | Text can't be scored (single endpoints only. Batch endpoints mark the item `rejected` instead) |
| 422 | `UNKNOWN_CATEGORY` | `details.suggestions` offers close matches (`ACOUNTANT` → `ACCOUNTANT`) |
| 500 | `INFERENCE_FAILED` / `INTERNAL_ERROR` | Model or server fault. The traceback is logged on the server and never sent to the client |
| 503 | `MODEL_NOT_READY` | Still starting up. Clients retry this automatically |

**Batch semantics:** a batch with one bad resume still returns 200. Every item comes
back in input order with its own `status`, and a `summary` gives the totals.
Only problems with the request as a whole (bad shape, too many items, duplicate ids) fail the batch.

### 5. Connecting the endpoint to Frontend / Backend squad apps
- **Frontend:** `clients/screening_client.js` has no dependencies. It works in the
  browser or Node 18+ and throws `ScreeningApiError` with the server's `code`. It also
  exports `ERROR_MESSAGES` (UI copy for each code). `demo/index.html` is a working page you can copy from.
  For TypeScript types: `npx openapi-typescript http://localhost:8000/openapi.json -o screening.d.ts`.
- **Backend:** `clients/screening_client.py` pools connections, retries on 503 and
  dropped connections, and splits large batches into server-sized chunks automatically.
- **CORS:** allowed origins come from `SCREENING_CORS_ORIGINS`. Defaults cover
  `localhost:3000` (React/Next) and `localhost:5173` (Vite).
- **Tracing:** send `X-Request-ID` and the API echoes it back and logs it, so one id
  can be followed across both services. Each response also includes `X-Process-Time-Ms`.
- **Auth:** set `SCREENING_API_KEY` to require an `X-API-Key` header on `/v1/*`.
  The probes stay open.
- **Docker:** see *Sprint 4 → Docker & deployment* below (`docker compose up --build` from the repo root).

## Sprint 4 — Team & Production

### Prediction logging
Every item scored by `/v1/match`, `/v1/match/batch` or `/v1/screen`, including rejected
ones, is written as one JSON line to `SCREENING_LOG_DIR/predictions-YYYY-MM-DD.jsonl`
(UTC, so files rotate daily). The last `DRIFT_WINDOW` records are also kept in memory.

```json
{"ts":"2026-10-24T10:54:19.437+00:00","request_id":"1eba76bf…","endpoint":"match","model":"all-MiniLM-L6-v2",
 "api_version":"1.1.0","build":"3f9c2ab","env":"render","latency_ms":3.75,"text_chars":10192,
 "text_sha":"0fceb9e8568ca41a","item_id":"0","status":"ok","best_category":"ACCOUNTANT","similarity":0.47,
 "cached":false,"top_categories":["ACCOUNTANT","FINANCE","BANKING"]}
```

* **No resume text, ever.** Only the length and a 16-hex-character SHA-256 prefix (enough to spot a
  resume sent twice, useless for recovering it). `test_monitoring.py` checks the log file for leaked text.
* `request_id` is the same one the client sent in `X-Request-ID`, so a Backend squad ticket can be
  traced straight to the logged prediction.
* Logging can't break a prediction. Write errors are caught, and an unwritable folder falls back to
  memory-only logging.
* On hosts with an ephemeral disk (Render, Railway), set `SCREENING_LOG_PREDICTIONS_STDOUT=true` so
  the records also go to the platform's log stream. With `SCREENING_LOG_FORMAT=json` (the Docker
  default), every log line is one JSON object.

### Drift monitoring
`GET /v1/monitoring/drift` compares the in-memory window with `artifacts/reference_profile.json`
(the training cohort as the deployed model sees it):

| check | warn / alert | catches |
|---|---|---|
| `similarity_psi` | PSI ≥ 0.10 / 0.25* | resumes less like any job we know |
| `low_confidence_rate_delta` | +10 / +20 points over the cohort's 16.7% | more out-of-scope applicants |
| `category_psi` | PSI ≥ 0.10 / 0.25* | a different mix of roles, or the model mis-routing |
| `text_length_psi` | PSI ≥ 0.10 / 0.25* | an upstream parser truncating or padding text |
| `rejection_rate` | 5% / 15% | a client sending empty or garbled documents |
| `embedding_centroid_cosine` | below what random cohort samples ever reach | the average *meaning* has moved |

\* On small windows the PSI cut-offs are raised to what sampling noise alone exceeds only 1% / 0.1%
of the time. Without that, 100 normal resumes over 24 categories already reach a PSI of about 0.2
and would page someone for nothing. Checks with fewer than `DRIFT_MIN_SAMPLES` items report
`insufficient_data`. For a longer period, run
`python scripts/drift_report.py --logs logs/ --days 7 --fail-on alert` (exit code 2 on alert).
Thresholds, the reasoning behind them and a runbook are in [`docs/MONITORING.md`](../docs/MONITORING.md).

### Metrics
`GET /metrics` is in Prometheus text format, with no extra dependency. It's unauthenticated
like the probes, and can be turned off with `SCREENING_METRICS_ENABLED=false`. Series:
`screening_http_requests_total{method,route,status}`, `screening_http_request_duration_seconds`
(histogram), `screening_predictions_total{endpoint,status}`, cache gauges, `screening_model_ready`,
`screening_build_info{version,build,env}`, and `screening_drift_status{check}` (0 ok, 1 warning,
2 alert, -1 not enough data). Routes are labelled by template, so random 404 paths don't create
new series.

### Docker & deployment
The image bakes in the model (`HF_HUB_OFFLINE=1`), copies `serving/artifacts/`, runs as a non-root
user, listens on `$PORT`, and reports healthy only once `/ready` returns 200. The same image runs on
Render (`render.yaml`), Railway (`railway.json`) and AWS ECS Express Mode
(`deploy/aws/deploy_ecs_express.sh`). Step-by-step instructions are in
[`docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md). After any deploy:

```bash
python scripts/smoke_test.py --url https://<host> --api-key $SCREENING_API_KEY --expect-model all-MiniLM-L6-v2
```

### Tests & CI
`.github/workflows/ci.yml` runs on every pull request: ruff, then pytest on Python 3.11 and 3.12,
then a Docker build with the real model and `smoke_test.py` against the running container. Pushes to
`main` also publish the image to GHCR. Besides the API tests, the suite covers the ML pipeline
(`test_pipeline.py`):
notebook portability, identical text cleaning in training and serving, bundle integrity, a
**top-1 ≥ 62% / top-3 ≥ 78% quality gate** on the committed embeddings (Sprint 2 measured 66.1% / 81.7%),
and drift checks that stay quiet on normal cohort traffic and fire on shifted traffic.
`test_deploy.py` keeps the Docker/Render/Railway/CI configs and this README in sync with the code.

## Configuration

| Env var | Default | Notes |
|---|---|---|
| `SCREENING_MODEL_NAME` | `all-MiniLM-L6-v2` | `hashing` = offline stand-in, for tests and demos only |
| `SCREENING_DEVICE` | `cpu` | `cuda` if you have a GPU |
| `SCREENING_ARTIFACTS_DIR` | `../data_pipeline/data/processed` if present, else `serving/artifacts` | `/data` in Docker |
| `SCREENING_SIMILARITY_FLOOR` | `0.35` | From `edge_cases.ipynb`. About 17% of the cohort's best matches fall below it, so tune it before relying on it |
| `SCREENING_MIN_TEXT_CHARS` / `MAX_TEXT_CHARS` | `30` / `50000` | |
| `SCREENING_MAX_BATCH_SIZE` | `64` | |
| `SCREENING_MAX_TOP_K` | `50` | Cap for `/v1/cohort/rank` |
| `SCREENING_EMBEDDING_CACHE_SIZE` | `4096` | About 1.5 KB per entry |
| `SCREENING_ENCODE_BATCH_SIZE` | `32` | Model mini-batch size |
| `SCREENING_CORS_ORIGINS` | localhost:3000, 5173 | Comma-separated |
| `SCREENING_API_KEY` | unset | |
| `SCREENING_LOG_DIR` | `serving/logs` (`/srv/logs` in Docker) | Prediction log folder. Empty = memory only |
| `SCREENING_LOG_PREDICTIONS_STDOUT` | `false` | Also print each prediction record to stdout (use on Render/Railway) |
| `SCREENING_LOG_FORMAT` | `text` (`json` in Docker) | `json` = one JSON object per log line |
| `SCREENING_DRIFT_WINDOW` | `1000` | Recent items kept in memory for `/v1/monitoring/drift` |
| `SCREENING_DRIFT_MIN_SAMPLES` | `100` | Below this a check reports `insufficient_data` |
| `SCREENING_METRICS_ENABLED` | `true` | Serve `/metrics` |
| `SCREENING_ENVIRONMENT` | `local` | Label in logs and metrics (`render`, `aws`, ...) |
| `SCREENING_GIT_SHA` | auto | Build id. Falls back to `RENDER_GIT_COMMIT` / `RAILWAY_GIT_COMMIT_SHA` / `GITHUB_SHA` |
| `PORT` | `8000` | Docker only. Set by the hosting platform |

## Known limitations
- Drift checks measure *change*, not *correctness*. There are still no hiring-outcome labels, so live
  accuracy can't be measured. Logging `request_id` lets outcomes be joined to predictions later.
- The drift window and metrics are per process. With several replicas, each reports its own traffic.
  Aggregate in Prometheus, or run `drift_report.py` over the combined log files.
- MiniLM reads only the first ~256 word pieces of each text, so most of a long resume
  is ignored. Sprint 1 embeddings had the same limit, so served scores match the offline
  evaluation. Chunking resumes and averaging the chunk embeddings is the obvious next step.
- The similarity floor isn't tuned against labelled "no match" data yet (the same caveat as in Sprint 2).
- The cache is per-process. With several workers or containers, each one keeps its own cache.
  Redis would make it shared.
- Single-resume requests that arrive at the same moment are not grouped into one model call.
  For heavy traffic, add a micro-batching queue in front of `encoder.encode`.
