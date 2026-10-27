## What & why
<!-- One or two sentences. Link the issue / sprint item: "Closes #12" -->

## Type
- [ ] Feature  - [ ] Bug fix  - [ ] Model / data change  - [ ] Docs  - [ ] Infra / CI

## Checklist
- [ ] `pytest` passes locally (`cd serving && pytest -q`)
- [ ] `ruff check serving` is clean
- [ ] New/changed endpoints are documented in `serving/README.md` (a test enforces this)
- [ ] **Model or data change?** Re-ran the notebooks, then `python serving/scripts/export_artifacts.py`,
      and committed `serving/artifacts/`. The quality gate in `tests/test_pipeline.py` still passes.
- [ ] **Breaking API change?** It goes under a new version (`/v2`), not into `/v1`.

## How I tested it
<!-- e.g. "ran smoke_test.py against docker compose up" -->
