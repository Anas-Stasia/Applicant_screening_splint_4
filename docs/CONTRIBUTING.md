# Team workflow — GitHub flow (Sprint 4)

`main` is always deployable, and merging to `main` **is** deploying. Everything else follows from that.

```
main ─────●──────────────●─────────────●──────►   (each ● = squash-merged PR → CI → deploy)
           \            /
            feature/drift-alerts
              commit · commit · push → PR → CI green + 1 review → squash-merge → delete branch
```

## The loop

1. **Branch from an up-to-date `main`.** Use short-lived branches named by intent:
   `feature/…`, `fix/…`, `model/…` (data or model changes), `docs/…`, `ci/…`.
   ```bash
   git switch main && git pull
   git switch -c feature/drift-alerts
   ```
2. **Commit small, push early.** Open a *draft* PR as soon as there's something to look at.
3. **Run the checks locally** (the same ones CI runs):
   ```bash
   cd serving
   pip install -r requirements-dev.txt
   ruff check . && pytest -q
   ```
4. **Open the PR.** The template asks the right questions: docs updated? artifacts re-exported? breaking change?
5. **CI must be green, plus one approving review** from someone on the other squad when the API
   contract changes.
6. **Squash-merge**, then delete the branch. CI runs again on `main`, publishes the image to GHCR, and
   Render (`checksPass`) or Railway (Wait for CI) deploys it.

## What CI checks (`.github/workflows/ci.yml`)

| Job | Runs | Fails the PR when |
|---|---|---|
| `lint` | `ruff check` | style or likely-bug findings (unused imports, bad comparisons, …) |
| `test` (3.11, 3.12) | `export_artifacts.py --check`, `pytest` | any API, monitoring, pipeline or config test fails |
| `docker` | builds the real image, starts it, runs `smoke_test.py` | the image doesn't build, doesn't become ready, or the real model gives a wrong answer |
| `deploy` (main only) | optional deploy hook + production smoke test | production doesn't pass the smoke test |

### Testing the ML pipeline, not just the code
`serving/tests/test_pipeline.py` is the part specific to ML. A PR fails if:
* a notebook gains a machine-specific path (`C:\Users\…`) or credentials, or stops using `config.ipynb`;
* the API's text cleaning stops matching `preprocess.ipynb` (training/serving skew);
* `serving/artifacts/` was edited by hand, or its embeddings are misaligned or not normalised;
* **the committed embeddings fall below top-1 62% / top-3 78%** (Sprint 2: 66.1% / 81.7%). This is the gate for model and data PRs;
* the drift monitor alarms on normal cohort traffic, or misses a shifted campaign.

## Model or data changes (`model/…` branches)
1. Re-run the notebooks (`main.ipynb`, then `main_sprint2.ipynb` to re-measure accuracy).
2. `python serving/scripts/export_artifacts.py`. This rebuilds the bundle, the drift reference and the manifest.
3. Commit `serving/artifacts/` together with the notebook changes, and put the new top-1/top-3 numbers in the PR description.
4. If the model name changes (`SCREENING_MODEL_NAME`), the Dockerfile's model download line must change too.

## Branch protection (repo admin, one-time)
GitHub → Settings → Branches → add a rule for `main`:
* Require a pull request before merging, with 1 approval
* Require status checks to pass: `lint`, `test (3.11)`, `test (3.12)`, `docker`
* Require branches to be up to date before merging
* Allow squash merging only (Settings → General)

## Releases
Tag milestones from `main` (`git tag -a v1.1.0 -m "Sprint 4" && git push --tags`). Bump
`API_VERSION` in `serving/app/main.py` for API changes: minor for additive changes, and a new
`/v2` path for breaking ones.

## Commit messages
Imperative and specific: `Add PSI drift endpoint`, `Fix TF-IDF notebook paths`.
Not: `splint 2`, `updates`.
