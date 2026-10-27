"""End-to-end tests for every endpoint, error path and caching behaviour."""
from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

from .conftest import ACCOUNTANT_RESUME, CHEF_RESUME, SAXOPHONIST, TEACHER_RESUME


# ----------------------------------------------------------------------- ops
def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"ready": True, "model": "hashing"}


def test_not_ready_returns_503(settings, encoder):
    app = create_app(settings=settings, encoder=encoder)
    c = TestClient(app)  # no `with` -> lifespan never runs -> model never loaded
    r = c.post("/v1/match", json={"text": ACCOUNTANT_RESUME})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "MODEL_NOT_READY"


def test_categories_and_model_info(client):
    body = client.get("/v1/categories").json()
    assert body == {"categories": ["ACCOUNTANT", "CHEF", "TEACHER"], "count": 3}
    info = client.get("/v1/model").json()
    assert info["model"] == "hashing" and info["cohort_size"] == 6
    assert info["limits"]["max_batch_size"] == 5


# --------------------------------------------------------------- single match
def test_match_single_ok(client):
    r = client.post("/v1/match", json={"text": ACCOUNTANT_RESUME, "top_k": 2})
    assert r.status_code == 200
    res = r.json()["result"]
    assert res["status"] == "ok"
    assert res["best_category"] == "ACCOUNTANT"
    assert len(res["top_matches"]) == 2
    assert res["top_matches"][0]["similarity"] >= res["top_matches"][1]["similarity"]


def test_match_strips_html(client):
    html = f"<div><p>{CHEF_RESUME}</p>&nbsp;</div>"
    assert client.post("/v1/match", json={"text": html}).json()["result"]["best_category"] == "CHEF"


@pytest.mark.parametrize("text,code", [
    ("", "EMPTY_TEXT"),
    ("   \n\t ", "EMPTY_TEXT"),
    ("Nurse.", "TEXT_TOO_SHORT"),
    ("!!!@@@###$$$%%%^^^&&&***()()()()", "NO_TEXT_CONTENT"),
    ("Experienced software engineer. " * 500, "TEXT_TOO_LONG"),
])
def test_match_rejects_unscoreable_text(client, text, code):
    r = client.post("/v1/match", json={"text": text})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == code
    assert err["request_id"]


def test_low_confidence_out_of_scope(client):
    res = client.post("/v1/match", json={"text": SAXOPHONIST}).json()["result"]
    assert res["status"] == "low_confidence"
    assert "manual review" in res["warnings"][0]


# ------------------------------------------------------------ schema errors
@pytest.mark.parametrize("payload,field", [
    ({}, "text"),
    ({"text": 123}, "text"),
    ({"text": ACCOUNTANT_RESUME, "top_k": 0}, "top_k"),
    ({"text": ACCOUNTANT_RESUME, "top_k": 99}, "top_k"),
    ({"txt": ACCOUNTANT_RESUME}, "txt"),  # unknown field -> rejected, not ignored
])
def test_validation_errors_use_envelope(client, payload, field):
    r = client.post("/v1/match", json=payload)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "VALIDATION_ERROR"
    assert any(d["field"] == field for d in err["details"])


def test_malformed_json(client):
    r = client.post("/v1/match", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_404_uses_envelope(client):
    r = client.get("/v1/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"


# --------------------------------------------------------------- batch match
def test_batch_mixed_items_keep_order(client):
    items = [
        {"id": "a", "text": ACCOUNTANT_RESUME},
        {"id": "b", "text": ""},
        {"id": "c", "text": CHEF_RESUME},
        {"id": "d", "text": SAXOPHONIST},
        {"text": TEACHER_RESUME},  # no id -> position used
    ]
    r = client.post("/v1/match/batch", json={"items": items})
    assert r.status_code == 200
    body = r.json()
    assert [x["id"] for x in body["results"]] == ["a", "b", "c", "d", "4"]
    assert [x["status"] for x in body["results"]] == ["ok", "rejected", "ok", "low_confidence", "ok"]
    assert body["results"][1]["warnings"][0].startswith("EMPTY_TEXT")
    assert body["results"][4]["best_category"] == "TEACHER"
    assert body["summary"] == {"total": 5, "ok": 3, "low_confidence": 1, "rejected": 1}


def test_batch_too_large(client):
    items = [{"text": ACCOUNTANT_RESUME}] * 6
    r = client.post("/v1/match/batch", json={"items": items})
    assert r.status_code == 413
    assert r.json()["error"]["details"] == {"limit": 5, "received": 6}


def test_batch_empty_and_duplicate_ids(client):
    assert client.post("/v1/match/batch", json={"items": []}).status_code == 422
    dup = [{"id": "x", "text": ACCOUNTANT_RESUME}, {"id": "x", "text": CHEF_RESUME}]
    r = client.post("/v1/match/batch", json={"items": dup})
    assert r.status_code == 422 and "Duplicate" in r.json()["error"]["details"][0]["message"]


# ------------------------------------------------------------------- caching
def test_cache_hit_skips_model(client, encoder):
    first = client.post("/v1/match", json={"text": TEACHER_RESUME}).json()["result"]
    calls = encoder.calls
    second = client.post("/v1/match", json={"text": "  " + TEACHER_RESUME + "  "}).json()["result"]
    assert first["cached"] is False and second["cached"] is True
    assert encoder.calls == calls  # no new forward pass
    assert first["similarity"] == second["similarity"]


def test_batch_encodes_once_and_dedupes(client, encoder):
    calls = encoder.calls
    items = [{"id": str(i), "text": f"{CHEF_RESUME} variant"} for i in range(4)]
    client.post("/v1/match/batch", json={"items": items})
    assert encoder.calls == calls + 1  # 4 identical texts -> one model call
    assert client.get("/v1/model").json()["cache"]["size"] >= 1


# ---------------------------------------------------------------- screening
def test_screen_by_category(client):
    cands = [{"id": "chef", "text": CHEF_RESUME}, {"id": "acct", "text": ACCOUNTANT_RESUME},
             {"id": "bad", "text": "hi"}, {"id": "teach", "text": TEACHER_RESUME}]
    body = client.post("/v1/screen", json={"job": {"category": "accountant"}, "candidates": cands}).json()
    assert body["job"] == "ACCOUNTANT"
    assert body["ranked"][0]["id"] == "acct" and body["ranked"][0]["rank"] == 1
    assert [c["rank"] for c in body["ranked"]] == [1, 2, 3]
    assert body["rejected"][0]["id"] == "bad" and body["rejected"][0]["rank"] is None


def test_screen_by_description_with_top_k(client):
    job = {"description": "Restaurant needs an experienced chef for kitchen, menu and catering work."}
    cands = [{"id": "acct", "text": ACCOUNTANT_RESUME}, {"id": "chef", "text": CHEF_RESUME}]
    body = client.post("/v1/screen", json={"job": job, "candidates": cands, "top_k": 1}).json()
    assert body["job"] == "custom" and [c["id"] for c in body["ranked"]] == ["chef"]


@pytest.mark.parametrize("job", [{}, {"category": "CHEF", "description": "both given here"}])
def test_screen_requires_exactly_one_job_field(client, job):
    r = client.post("/v1/screen", json={"job": job, "candidates": [{"text": CHEF_RESUME}]})
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_unknown_category_suggests(client):
    r = client.post("/v1/screen", json={"job": {"category": "ACCOUNTANTS"},
                                        "candidates": [{"text": CHEF_RESUME}]})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "UNKNOWN_CATEGORY" and err["details"]["suggestions"] == ["ACCOUNTANT"]


def test_short_job_description_rejected(client):
    r = client.post("/v1/screen", json={"job": {"description": "chef"}, "candidates": [{"text": CHEF_RESUME}]})
    assert r.status_code == 422 and r.json()["error"]["code"] == "TEXT_TOO_SHORT"


# --------------------------------------------------------------- cohort rank
def test_cohort_rank(client):
    body = client.post("/v1/cohort/rank", json={"job": {"category": "TEACHER"}, "top_k": 2}).json()
    assert body["cohort_size"] == 6
    assert [m["rank"] for m in body["matches"]] == [1, 2]
    assert {m["category"] for m in body["matches"]} == {"TEACHER"}
    assert body["matches"][0]["similarity"] >= body["matches"][1]["similarity"]


# ---------------------------------------------------------------- integration
def test_request_id_and_timing_headers(client):
    r = client.get("/health", headers={"X-Request-ID": "trace-123"})
    assert r.headers["X-Request-ID"] == "trace-123"
    assert float(r.headers["X-Process-Time-Ms"]) >= 0


def test_cors_preflight(client):
    r = client.options("/v1/match", headers={
        "Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_api_key(settings, encoder):
    app = create_app(settings=replace(settings, api_key="s3cret"), encoder=encoder)
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200  # probes stay open
        r = c.get("/v1/categories")
        assert r.status_code == 401 and r.json()["error"]["code"] == "UNAUTHORIZED"
        assert c.get("/v1/categories", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_model_failure_is_500_envelope(settings, encoder):
    app = create_app(settings=settings, encoder=encoder)
    with TestClient(app, raise_server_exceptions=False) as c:
        def boom(*a, **k):
            raise RuntimeError("CUDA out of memory")
        encoder.encode = boom
        r = c.post("/v1/match", json={"text": ACCOUNTANT_RESUME + " new text"})
        assert r.status_code == 500
        err = r.json()["error"]
        assert err["code"] == "INFERENCE_FAILED"
        assert "CUDA" not in err["message"]  # internals are logged, not leaked to clients


def test_openapi_documents_error_schema(client):
    spec = client.get("/openapi.json").json()
    assert "ErrorResponse" in spec["components"]["schemas"]
    assert "422" in spec["paths"]["/v1/match/batch"]["post"]["responses"]


# ------------------------------------------------- reuse of saved embeddings
def test_saved_embeddings_are_reused_only_if_they_match_the_model(settings, encoder, artifacts):
    import numpy as np
    import pandas as pd

    from app.encoders import HashingEncoder

    emb = artifacts / "embeddings"
    emb.mkdir()
    jobs = pd.read_csv(artifacts / "job_descriptions.csv")
    resumes = pd.read_csv(artifacts / "resumes_clean.csv")
    np.save(emb / "job_embeddings.npy", encoder.encode(jobs["job_description"].tolist()))
    np.save(emb / "resume_embeddings.npy", encoder.encode(resumes["clean_text"].tolist()))
    jobs[["Category"]].to_csv(emb / "job_index.csv", index=False)
    resumes[["ID", "Category"]].to_csv(emb / "resume_index.csv", index=False)

    with TestClient(create_app(settings=settings, encoder=HashingEncoder(dim=256))) as c:
        info = c.get("/v1/model").json()
        assert info["job_embeddings_source"] == "loaded job_embeddings.npy"
        assert info["cohort_embeddings_source"] == "loaded resume_embeddings.npy"

    # Same shape, different model (simulated by a different hash seed) -> re-encode.
    class OtherModel(HashingEncoder):
        def encode(self, texts, batch_size=32):
            return super().encode([t[::-1] for t in texts], batch_size)

    with TestClient(create_app(settings=settings, encoder=OtherModel(dim=256))) as c:
        assert c.get("/v1/model").json()["job_embeddings_source"] == "encoded at startup"
