"""Test fixtures.

Tests build a tiny synthetic artifacts folder and use the dependency-free
``HashingEncoder``, so the suite runs in a couple of seconds with no model
download and no dependency on the (gitignored) data folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `app` importable

from app.encoders import HashingEncoder  # noqa: E402
from app.main import create_app  # noqa: E402
from app.settings import Settings  # noqa: E402

JOBS = {
    "ACCOUNTANT": "Looking for a ACCOUNTANT candidate with experience in: accounting, tax, payroll, "
                  "ledger, reconciliation, audit, financial statements, invoices, budgeting",
    "CHEF": "Looking for a CHEF candidate with experience in: kitchen, cooking, menu, recipes, "
            "food safety, catering, restaurant, pastry, culinary",
    "TEACHER": "Looking for a TEACHER candidate with experience in: classroom, students, lesson plans, "
               "curriculum, teaching, grading, education, school, tutoring",
}

ACCOUNTANT_RESUME = ("Senior accountant handling tax filings, payroll, general ledger reconciliation, "
                     "audit preparation, financial statements and monthly budgeting for clients.")
CHEF_RESUME = ("Head chef running a busy restaurant kitchen: menu design, recipes, food safety, "
               "catering events, pastry and culinary team training.")
TEACHER_RESUME = ("High school teacher creating lesson plans and curriculum, grading students, "
                  "classroom management, tutoring and education outreach at the school.")
SAXOPHONIST = ("Professional saxophonist with ten years performing jazz concerts, touring "
               "internationally, and composing music for orchestras.")


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    pd.DataFrame({"Category": list(JOBS), "job_description": list(JOBS.values())}).to_csv(
        tmp_path / "job_descriptions.csv", index=False)
    pd.DataFrame({
        "ID": [101, 102, 103, 104, 105, 106],
        "Category": ["ACCOUNTANT", "ACCOUNTANT", "CHEF", "CHEF", "TEACHER", "TEACHER"],
        "clean_text": [ACCOUNTANT_RESUME, "Tax accountant, payroll and audit, ledger and invoices.",
                       CHEF_RESUME, "Pastry chef, kitchen recipes, menu and catering.",
                       TEACHER_RESUME, "Teacher of students, classroom lesson plans and grading."],
    }).to_csv(tmp_path / "resumes_clean.csv", index=False)
    return tmp_path


@pytest.fixture
def settings(artifacts: Path) -> Settings:
    return Settings(artifacts_dir=artifacts, similarity_floor=0.2, max_batch_size=5,
                    max_text_chars=5000, embedding_cache_size=100, api_key=None,
                    log_dir=artifacts / "logs", drift_min_samples=5, drift_window=200)


@pytest.fixture
def encoder() -> HashingEncoder:
    return HashingEncoder(dim=256)


@pytest.fixture
def client(settings, encoder):
    app = create_app(settings=settings, encoder=encoder)
    with TestClient(app, raise_server_exceptions=False) as c:  # `with` runs the lifespan (model load)
        yield c
