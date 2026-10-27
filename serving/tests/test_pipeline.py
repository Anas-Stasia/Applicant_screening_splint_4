"""Sprint 4: automated tests for the ML pipeline itself (not just the API).

These run in CI on every pull request. They guard the things that silently break
an ML system without failing any API test:

* the notebooks stay portable (no machine-specific paths, no credentials);
* the API cleans text exactly like the training pipeline did (no training/serving skew);
* the committed artifact bundle is intact, aligned and normalised;
* the model artifacts still hit the accuracy measured in Sprint 2 (a quality gate);
* the drift monitor stays quiet on normal traffic and fires on shifted traffic.

Tests that need the full (gitignored) pipeline data skip themselves when it's absent.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SERVING = Path(__file__).resolve().parents[1]
PROJECT = SERVING.parent
PIPELINE = PROJECT / "data_pipeline"
BUNDLE = SERVING / "artifacts"
RAW_CSV = PIPELINE / "data" / "raw" / "Resume.csv"

sys.path.insert(0, str(SERVING / "scripts"))
from app.monitoring import DriftMonitor, ReferenceProfile  # noqa: E402
from app.text import clean_text  # noqa: E402

NOTEBOOKS = sorted(PROJECT.glob("*.ipynb")) + sorted(PIPELINE.glob("*.ipynb"))
ORCHESTRATORS = {"config.ipynb", "main.ipynb", "main_sprint2.ipynb"}


def _code_cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


# ------------------------------------------------------------- notebook hygiene
def test_notebooks_exist():
    names = {p.name for p in NOTEBOOKS}
    assert {"config.ipynb", "preprocess.ipynb", "embeddings.ipynb", "tfidf_baseline.ipynb",
            "evaluate_recommendations.ipynb", "sprint3_serving.ipynb", "sprint4_production.ipynb"} <= names


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid(path):
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.read(path, as_version=4))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_portable(path):
    """The Sprint 1 bug: a notebook with C:\\Users\\... paths runs on one laptop only."""
    assert path.parent == PIPELINE, f"{path.name} must live in data_pipeline/ next to config.ipynb"
    for src in _code_cells(path):
        assert not re.search(r"\b[A-Za-z]:\\(Users|Documents|Downloads)", src), f"Windows path in {path.name}"
        assert not re.search(r"(/Users/|/home/)\w+/", src), f"absolute home path in {path.name}"
    if path.name not in ORCHESTRATORS:
        assert any("%run config.ipynb" in s for s in _code_cells(path)), f"{path.name} must %run config.ipynb"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_credentials(path):
    text = path.read_text(encoding="utf-8")
    assert not re.search(r'"key"\s*:\s*"[0-9a-f]{32}"', text), "Kaggle API key committed"
    assert not re.search(r"KGAT_[A-Za-z0-9]{20,}", text), "Kaggle access token committed"


# ---------------------------------------------------- training/serving parity
def _notebook_clean_text():
    """Load clean_text() from preprocess.ipynb itself — functions and imports only."""
    pytest.importorskip("bs4")
    src = next(s for s in _code_cells(PIPELINE / "preprocess.ipynb") if "def clean_text" in s)
    tree = ast.parse(src)
    tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    ns: dict = {}
    exec(compile(tree, "preprocess.ipynb", "exec"), ns)
    return ns["clean_text"]


PARITY_SAMPLES = [
    "Plain resume text with   extra   spaces\nand newlines.",
    "<div><h1>Accountant</h1><p>Payroll &amp; tax</p></div>",
    "Malformed entity &ndash dash and a stray <15> bracket",
    "Café manager — résumé with “smart quotes” and emoji 🎉",
    "<br/>Line<br>break<p>para</p>&nbsp;&nbsp;nbsp",
    "A < B and C > D, R&D department, AT&T",
    "",
    "\t\n  ",
]


@pytest.mark.parametrize("text", PARITY_SAMPLES)
def test_serving_cleaner_matches_notebook(text):
    assert clean_text(text) == _notebook_clean_text()(text)


@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw Kaggle data not downloaded")
def test_serving_cleaner_matches_notebook_on_full_dataset():
    ref = _notebook_clean_text()
    raw = pd.read_csv(RAW_CSV)
    texts = list(raw["Resume_str"].astype(str)) + list(raw["Resume_html"].astype(str))
    diffs = [i for i, t in enumerate(texts) if ref(t) != clean_text(t)]
    assert not diffs, f"{len(diffs)} of {len(texts)} texts are cleaned differently, e.g. #{diffs[:5]}"


# ------------------------------------------------------------ artifact bundle
@pytest.fixture(scope="module")
def bundle():
    if not (BUNDLE / "manifest.json").exists():
        pytest.skip("serving/artifacts not exported")
    jobs = pd.read_csv(BUNDLE / "job_descriptions.csv")
    resumes = pd.read_csv(BUNDLE / "resumes_clean.csv.gz")
    return {
        "manifest": json.loads((BUNDLE / "manifest.json").read_text()),
        "jobs": jobs,
        "resumes": resumes,
        "J": np.load(BUNDLE / "embeddings" / "job_embeddings.npy"),
        "R": np.load(BUNDLE / "embeddings" / "resume_embeddings.npy"),
        "job_index": pd.read_csv(BUNDLE / "embeddings" / "job_index.csv"),
        "resume_index": pd.read_csv(BUNDLE / "embeddings" / "resume_index.csv"),
        "profile": ReferenceProfile.from_json((BUNDLE / "reference_profile.json").read_text()),
    }


def test_bundle_matches_manifest(bundle):
    from export_artifacts import check
    assert check(BUNDLE) == [], "bundle changed without re-running scripts/export_artifacts.py"


def test_bundle_is_aligned_and_normalised(bundle):
    b = bundle
    assert b["J"].shape == (len(b["jobs"]), 384) and b["R"].shape == (len(b["resumes"]), 384)
    assert b["job_index"]["Category"].tolist() == b["jobs"]["Category"].tolist()
    assert b["resume_index"]["ID"].astype(str).tolist() == b["resumes"]["ID"].astype(str).tolist()
    assert np.allclose(np.linalg.norm(b["J"], axis=1), 1, atol=1e-3)
    assert np.allclose(np.linalg.norm(b["R"], axis=1), 1, atol=1e-3)
    assert not np.isnan(b["R"]).any()
    assert b["manifest"]["resumes"] == len(b["resumes"]) == 2481
    assert set(b["resumes"]["Category"]) == set(b["jobs"]["Category"])


def test_bundle_text_is_long_enough_for_the_model(bundle):
    # MiniLM reads ~256 word pieces (~1,100 chars). Truncated texts must keep at least
    # that much, or the startup spot check / any re-encode would see different input.
    lengths = bundle["resumes"]["clean_text"].astype(str).str.len()
    assert bundle["manifest"]["max_chars"] >= 1500
    assert lengths.max() <= bundle["manifest"]["max_chars"]
    assert lengths.min() >= 200          # preprocess.ipynb already dropped anything shorter
    assert lengths.median() >= 2000


def test_reference_profile_matches_bundle(bundle):
    p, b = bundle["profile"], bundle
    rebuilt = ReferenceProfile.build(
        model=p.model, categories=b["jobs"]["Category"].tolist(), job_embeddings=b["J"],
        cohort_embeddings=b["R"], text_lengths=np.ones(len(b["R"])), similarity_floor=p.similarity_floor)
    assert rebuilt.similarity_probs == pytest.approx(p.similarity_probs, abs=1e-6)
    assert rebuilt.category_probs == pytest.approx(p.category_probs, abs=1e-6)
    assert rebuilt.low_confidence_rate == pytest.approx(p.low_confidence_rate, abs=1e-6)


# ------------------------------------------------------------- quality gate
def test_model_quality_gate(bundle):
    """Sprint 2 measured 66.1% top-1 / 81.7% top-3 on the full cohort. If a re-export
    (new model, new job descriptions, a preprocessing change) drops well below that,
    the pull request fails instead of shipping a worse model."""
    b = bundle
    sims = b["R"] @ b["J"].T
    cats = np.asarray(b["jobs"]["Category"])
    truth = b["resumes"]["Category"].to_numpy()
    top1 = float((cats[sims.argmax(1)] == truth).mean())
    top3 = float(np.mean([t in cats[np.argsort(-row)[:3]] for t, row in zip(truth, sims)]))
    assert top1 >= 0.62, f"top-1 accuracy fell to {top1:.1%}"
    assert top3 >= 0.78, f"top-3 accuracy fell to {top3:.1%}"
    assert b["profile"].low_confidence_rate <= 0.25


# ---------------------------------------------------------- drift behaviour
def _records(b, idx):
    sims = b["R"][idx] @ b["J"].T
    cats = np.asarray(b["jobs"]["Category"])
    lengths = np.full(len(idx), 5500)  # bundle text is truncated; use a typical full length
    return [{"ts": "t", "endpoint": "match", "status": "ok", "similarity": float(s.max()),
             "best_category": str(cats[s.argmax()]), "text_chars": int(n)} for s, n in zip(sims, lengths)]


def test_drift_quiet_on_normal_traffic(bundle):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(bundle["R"]), 500, replace=False)
    report = DriftMonitor(bundle["profile"], min_samples=100).report(_records(bundle, idx), bundle["R"][idx])
    by = {c["name"]: c["status"] for c in report["checks"]}
    for name in ("similarity_psi", "category_psi", "low_confidence_rate_delta", "embedding_centroid_cosine"):
        assert by[name] == "ok", (name, report["checks"])


def test_drift_fires_on_shifted_traffic(bundle):
    # A hiring campaign for weak, under-represented roles (ARTS/APPAREL/AUTOMOBILE).
    mask = bundle["resumes"]["Category"].isin(["ARTS", "APPAREL", "AUTOMOBILE"]).to_numpy()
    idx = np.where(mask)[0]
    report = DriftMonitor(bundle["profile"], min_samples=100).report(_records(bundle, idx), bundle["R"][idx])
    by = {c["name"]: c["status"] for c in report["checks"]}
    assert report["status"] == "alert"
    assert by["category_psi"] == "alert" and by["similarity_psi"] == "alert"
