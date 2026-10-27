# Monitoring guide (Sprint 4)

What the API records, how drift is measured, and what to do when an alert fires.

## 1. What gets logged

| Where | What | Kept |
|---|---|---|
| `SCREENING_LOG_DIR/predictions-YYYY-MM-DD.jsonl` | one line per scored/rejected resume | until you delete it (mount a volume in Docker) |
| stdout (`screening.predictions` logger) | the same records, if `SCREENING_LOG_PREDICTIONS_STDOUT=true` | by the platform (Render/Railway logs, CloudWatch) |
| stdout (`screening.api`, `screening.service`) | request lines, startup, errors with `request_id` | by the platform |
| memory | last `SCREENING_DRIFT_WINDOW` records + their embeddings | until restart |

Fields: `ts, request_id, endpoint, model, api_version, build, env, latency_ms, item_id, status,
best_category, similarity, top_categories, cached, text_chars, text_sha, reject_code`, plus
`job` and `rank` for `/v1/screen`.

**Privacy rule: resume text is never logged.** `text_sha` is the first 16 hex characters of the
SHA-256 of the cleaned text. It identifies repeats, and it can't be reversed into the resume.
`tests/test_monitoring.py::test_every_scored_item_is_logged_without_text` enforces this.

## 2. How drift is measured

There are no ground-truth labels in production, so we watch inputs and model confidence and
compare them with the **reference profile**: the 2,481-resume training cohort as the deployed
model sees it (`serving/artifacts/reference_profile.json`, written by `export_artifacts.py`).

### Population Stability Index (PSI)
For a histogram with bins *i*, reference share *eᵢ* and recent share *aᵢ*:

  PSI = Σ (aᵢ − eᵢ) · ln(aᵢ / eᵢ)

Similarity and text length use the cohort's **deciles** as bins (10 bins, 10% each). Categories
use the 24 predicted categories. Usual reading: < 0.10 stable, 0.10–0.25 moderate, > 0.25 large.

### Why the thresholds move with window size
Those cut-offs assume large samples. Sampling 100 resumes *from the cohort itself* gives a
category PSI around 0.2 just from noise. We measured this: with fixed cut-offs, 100 normal resumes
triggered a category warning or alert in nearly every one of 200 random draws. So for each check
the API simulates 2,000 samples of the current window size from the reference and raises
`warn_at` / `alert_at` to the 99th / 99.9th percentile of that noise, whenever that's higher than
0.10 / 0.25. Measured on the real cohort embeddings:

| window of normal cohort resumes | false alarms with fixed cut-offs (any check, 200 draws) | with noise-aware cut-offs |
|---|---|---|
| 100 | category PSI fired in 199 of 200 draws | at most 3 of 200 for any single check |
| 300 | category PSI warned in 26 of 200 | at most 2 of 200 |
| 1,000 | none | none (the cut-offs are back at 0.10 / 0.25) |

Real shifts are still caught. A 100-resume campaign drawn from ARTS, APPAREL and AUTOMOBILE
gives a category PSI of 1.52 and a similarity PSI of 0.52, both alerts.
`tests/test_pipeline.py` re-checks both behaviours on the committed embeddings.

### The six checks

| check | reference | warn | alert |
|---|---|---|---|
| `similarity_psi` | cohort best-category similarity deciles | 0.10* | 0.25* |
| `low_confidence_rate_delta` | 16.7% of the cohort is below 0.35 | +0.10 | +0.20 |
| `category_psi` | cohort predicted-category mix | 0.10* | 0.25* |
| `text_length_psi` | cohort cleaned-text length deciles | 0.10* | 0.25* |
| `rejection_rate` | 0% (the cohort was all valid) | 5% | 15% |
| `embedding_centroid_cosine` | the lowest cosine reached by 200 random cohort samples of 100 (0.986) | below it | 0.05 below it |

\* raised on small windows, as described above. `similarity_psi`, `low_confidence_rate_delta` and
`category_psi` only use `/v1/match` and `/v1/match/batch` traffic. `/v1/screen` scores against a
job the caller picked, so its similarities aren't comparable to the cohort's best-category scores.

## 3. Where to look

* **Live:** `GET /v1/monitoring/drift` (the in-memory window) and `GET /v1/monitoring/predictions?limit=50`.
* **Dashboards and alerts:** scrape `GET /metrics`. Example Prometheus alert rules:
  ```yaml
  groups:
    - name: screening
      rules:
        - alert: ScreeningDrift
          expr: max(screening_drift_status) >= 2
          for: 30m
          annotations: {summary: "Input drift alert on {{ $labels.check }}"}
        - alert: ScreeningErrors
          expr: sum(rate(screening_http_requests_total{status=~"5.."}[5m])) > 0.05
          for: 10m
        - alert: ScreeningSlow
          expr: histogram_quantile(0.95, sum by (le) (rate(screening_http_request_duration_seconds_bucket{route="/v1/screen"}[5m]))) > 1
          for: 10m
  ```
* **Weekly review:** `python serving/scripts/drift_report.py --logs <folder> --days 7 --json drift.json`.
  Add `--fail-on alert` in a cron job or scheduled CI run to get exit code 2 when there's an alert.

## 4. Runbook: an alert fired

| Alert | First question | Likely causes | Action |
|---|---|---|---|
| `rejection_rate` | Which `reject_code`, and which client? | A broken upload or PDF parser in a squad's app sends empty or garbled text | Filter `/v1/monitoring/predictions` by time and `reject_code`, trace the `request_id` to the client, and tell its owners |
| `text_length_psi` | Shorter or longer than usual? | Truncation (a form field limit), or a parser that now includes headers and footers | Compare `text_chars` before and after the deploy (`build` field) on either side |
| `category_psi` alone | Which categories grew? (`detail` shows the top 3) | A real hiring campaign. This is often expected | If expected, note it. If the model is routing everything to one category, check for a bad `job_descriptions.csv` or model change |
| `similarity_psi` + `low_confidence_rate_delta` | Are the new resumes out of scope? | Roles outside the 24 categories (the saxophonist case), or a non-English source | Review a sample of `low_confidence` items by hand. Consider new categories, or tune `SIMILARITY_FLOOR` |
| `embedding_centroid_cosine` | Did anything else fire? | A broad change in the population | Look at it together with the other checks. On its own it's a prompt to look, not a diagnosis |

After a confirmed, lasting shift (new categories, new data sources): re-run the Sprint 1/2
notebooks on the new data and run `export_artifacts.py`, which also rebuilds the reference
profile. Then open a PR. CI's quality gate checks the new artifacts before they ship.

## 5. Limitations
* Drift means *change*, not *wrong*. Rankings can degrade without drift, and drift can be harmless.
  Measuring accuracy needs outcome labels (who was shortlisted or hired), joined to predictions by `request_id`.
* The window and metrics are per process. With several replicas, aggregate in Prometheus
  (`max`/`sum`), or run `drift_report.py` over all replicas' log files.
* The offline report has no embeddings, so its centroid check reports `insufficient_data`.
