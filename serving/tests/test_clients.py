"""The Python client the Backend squad will use, exercised against the in-process app."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clients"))
from screening_client import ScreeningApiError, ScreeningClient  # noqa: E402

from .conftest import ACCOUNTANT_RESUME, CHEF_RESUME, TEACHER_RESUME


def test_python_client_end_to_end(client):
    api = ScreeningClient(http=client, max_batch_size=2)  # force chunking (server limit is 5)
    assert api.health() == {"status": "ok"}
    assert "CHEF" in api.categories()
    assert api.match(CHEF_RESUME)["best_category"] == "CHEF"

    items = [{"id": f"r{i}", "text": t} for i, t in
             enumerate([ACCOUNTANT_RESUME, CHEF_RESUME, TEACHER_RESUME, "", CHEF_RESUME + " again"])]
    results = api.match_batch(items)  # 5 items -> 3 requests of <=2
    assert [r["id"] for r in results] == ["r0", "r1", "r2", "r3", "r4"]
    assert results[3]["status"] == "rejected"

    ranked = api.screen({"category": "TEACHER"}, [{"id": "t", "text": TEACHER_RESUME},
                                                  {"id": "c", "text": CHEF_RESUME}])
    assert ranked["ranked"][0]["id"] == "t"


def test_python_client_raises_typed_error(client):
    api = ScreeningClient(http=client)
    with pytest.raises(ScreeningApiError) as exc:
        api.match("too short")
    assert exc.value.status == 422 and exc.value.code == "TEXT_TOO_SHORT" and exc.value.request_id


def test_demo_page_served(client):
    r = client.get("/demo/")
    assert r.status_code == 200 and "Applicant Screening API" in r.text
