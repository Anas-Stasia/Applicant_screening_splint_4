"""Sprint 4: prediction logging, drift monitoring and metrics."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.monitoring import DriftMonitor, PredictionLog, ReferenceProfile, psi, psi_thresholds

from .conftest import ACCOUNTANT_RESUME, CHEF_RESUME, SAXOPHONIST, TEACHER_RESUME


# ------------------------------------------------------------------- PSI maths
def test_psi_identical_is_zero_and_shift_is_large():
    ref = [0.1] * 10
    assert psi(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert psi(ref, [100] * 10) == pytest.approx(0.0, abs=1e-9)  # raw counts are normalised
    shifted = [0.0] * 5 + [0.2] * 5                               # half the bins emptied
    assert psi(ref, shifted) > 0.25


def test_psi_thresholds_scale_with_window_size():
    probs = [1 / 24] * 24
    small_warn, small_alert = psi_thresholds(probs, 100)
    big_warn, big_alert = psi_thresholds(probs, 5000)
    assert small_warn > 0.10 and small_alert > small_warn          # noise dominates on 100 samples
    assert (big_warn, big_alert) == (0.10, 0.25)                    # rule of thumb on large windows
    # Under the null (samples drawn from the reference) the warn level is exceeded ~1% of the time.
    rng = np.random.default_rng(3)
    fp = np.mean([psi(probs, rng.multinomial(100, probs)) > small_warn for _ in range(1000)])
    assert fp < 0.03


def test_psi_empty_bin_is_finite():
    assert np.isfinite(psi([0.5, 0.5, 0.0], [0.0, 0.5, 0.5]))


# -------------------------------------------------------------- reference profile
@pytest.fixture
def profile() -> ReferenceProfile:
    rng = np.random.default_rng(0)
    jobs = rng.normal(size=(3, 16)).astype(np.float32)
    jobs /= np.linalg.norm(jobs, axis=1, keepdims=True)
    # cohort = noisy copies of the job vectors -> realistic "best similarity" spread
    cohort = np.repeat(jobs, 100, axis=0) + rng.normal(scale=0.6, size=(300, 16))
    cohort = (cohort / np.linalg.norm(cohort, axis=1, keepdims=True)).astype(np.float32)
    return ReferenceProfile.build(model="test", categories=["A", "B", "C"], job_embeddings=jobs,
                                  cohort_embeddings=cohort, text_lengths=rng.integers(500, 9000, 300),
                                  similarity_floor=0.35, sample_size=50)


def test_profile_shape_and_json_roundtrip(profile):
    assert sum(profile.similarity_probs) == pytest.approx(1.0, abs=1e-4)
    assert sum(profile.category_probs.values()) == pytest.approx(1.0, abs=1e-4)
    assert profile.similarity_edges[0] == float("-inf") and profile.similarity_edges[-1] == float("inf")
    back = ReferenceProfile.from_json(profile.to_json())
    assert back.similarity_edges == profile.similarity_edges
    assert back.length_probs == profile.length_probs
    assert back.source == "loaded"
    json.loads(profile.to_json())  # valid JSON despite the infinite bin edges


def _records_from(sims, cats, lengths, endpoint="match"):
    return [{"ts": f"2026-10-24T00:00:{i % 60:02d}", "endpoint": endpoint, "status": "ok",
             "similarity": float(s), "best_category": c, "text_chars": int(n)}
            for i, (s, c, n) in enumerate(zip(sims, cats, lengths))]


def test_drift_ok_on_reference_like_traffic(profile):
    rng = np.random.default_rng(1)
    # resample values that fall inside the reference distribution's bins
    edges = [e for e in profile.similarity_edges[1:-1]]
    sims = rng.choice(edges, 400) + 1e-4
    cats = rng.choice(list(profile.category_probs), 400, p=list(profile.category_probs.values()))
    lengths = rng.integers(500, 9000, 400)
    report = DriftMonitor(profile, min_samples=50).report(_records_from(sims, cats, lengths))
    by = {c["name"]: c for c in report["checks"]}
    assert by["category_psi"]["status"] == "ok"
    assert by["text_length_psi"]["status"] == "ok"
    assert by["rejection_rate"]["status"] == "ok"


def test_drift_alerts_on_shifted_traffic(profile):
    n = 300
    records = _records_from([0.05] * n, ["A"] * n, [80] * n)  # tiny texts, all one class, no confidence
    records += [{"ts": "2026-10-24T01:00:00", "endpoint": "match", "status": "rejected"}] * 100
    report = DriftMonitor(profile, min_samples=50).report(records)
    by = {c["name"]: c for c in report["checks"]}
    assert report["status"] == "alert"
    for name in ("similarity_psi", "category_psi", "text_length_psi", "low_confidence_rate_delta",
                 "rejection_rate"):
        assert by[name]["status"] == "alert", name


def test_drift_insufficient_data(profile):
    report = DriftMonitor(profile, min_samples=100).report(_records_from([0.4] * 3, ["A"] * 3, [900] * 3))
    assert report["status"] == "insufficient_data"
    assert all(c["status"] == "insufficient_data" for c in report["checks"])


def test_centroid_check_detects_semantic_shift(profile):
    monitor = DriftMonitor(profile, min_samples=10)
    same = np.tile(np.asarray(profile.centroid, dtype=np.float32), (50, 1))
    other = -same
    ok = {c["name"]: c for c in monitor.report([], same)["checks"]}["embedding_centroid_cosine"]
    bad = {c["name"]: c for c in monitor.report([], other)["checks"]}["embedding_centroid_cosine"]
    assert ok["status"] == "ok" and bad["status"] == "alert"


# -------------------------------------------------------------- prediction log
def test_prediction_log_writes_jsonl_and_bounds_window(tmp_path):
    log = PredictionLog(tmp_path / "logs", window=3)
    log.record([{"endpoint": "match", "status": "ok", "i": i} for i in range(5)])
    assert [r["i"] for r in log.recent()] == [2, 3, 4]          # in-memory window is bounded
    lines = log.current_file().read_text().splitlines()
    assert len(lines) == 5                                        # the file keeps everything
    assert PredictionLog.read_files([log.current_file()])[0]["i"] == 0


def test_prediction_log_survives_unwritable_dir(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    log = PredictionLog(blocker / "logs")  # parent is a file -> mkdir fails
    assert log.log_dir is None
    log.record([{"endpoint": "match", "status": "ok"}])          # still works, memory only
    assert log.total == 1


# ------------------------------------------------------------------- API level
def test_every_scored_item_is_logged_without_text(client, settings):
    client.post("/v1/match", json={"text": ACCOUNTANT_RESUME}, headers={"X-Request-ID": "trace-1"})
    client.post("/v1/match/batch", json={"items": [{"id": "a", "text": CHEF_RESUME}, {"id": "b", "text": ""}]})
    client.post("/v1/screen", json={"job": {"category": "TEACHER"},
                                    "candidates": [{"id": "t", "text": TEACHER_RESUME}, {"id": "x", "text": "hi"}]})
    body = client.get("/v1/monitoring/predictions", params={"limit": 100}).json()
    recs = body["records"]
    assert body["count"] == 5 and body["total_logged"] == 5
    assert [r["endpoint"] for r in recs] == ["match", "match_batch", "match_batch", "screen", "screen"]
    assert recs[0]["request_id"] == "trace-1" and recs[0]["best_category"] == "ACCOUNTANT"
    assert {r["item_id"]: r.get("reject_code") for r in recs if r["status"] == "rejected"} == {
        "b": "EMPTY_TEXT", "x": "TEXT_TOO_SHORT"}
    assert recs[3]["job"] == "TEACHER" and recs[3]["rank"] == 1

    # The log on disk must never contain resume text.
    log_file = next((settings.log_dir).glob("predictions-*.jsonl"))
    raw = log_file.read_text()
    assert len(raw.splitlines()) == 5
    for text in (ACCOUNTANT_RESUME, CHEF_RESUME, TEACHER_RESUME):
        assert text[:40] not in raw
    assert all(len(r["text_sha"]) == 16 for r in recs if r["text_chars"])


def test_single_match_rejection_is_logged(client):
    assert client.post("/v1/match", json={"text": "Nurse."}).status_code == 422
    rec = client.get("/v1/monitoring/predictions").json()["records"][-1]
    assert rec["status"] == "rejected" and rec["reject_code"] == "TEXT_TOO_SHORT"


def test_endpoint_filter(client):
    client.post("/v1/match", json={"text": ACCOUNTANT_RESUME})
    client.post("/v1/screen", json={"job": {"category": "CHEF"}, "candidates": [{"text": CHEF_RESUME}]})
    only = client.get("/v1/monitoring/predictions", params={"endpoint": "screen"}).json()["records"]
    assert [r["endpoint"] for r in only] == ["screen"]


def test_drift_endpoint(client):
    r = client.get("/v1/monitoring/drift")
    assert r.status_code == 200 and r.json()["status"] == "insufficient_data"
    # 3 categories, mostly out-of-scope traffic -> confidence checks should fire
    items = [{"id": str(i), "text": f"{SAXOPHONIST} gig number {i}"} for i in range(5)]
    for _ in range(2):
        client.post("/v1/match/batch", json={"items": items})
    report = client.get("/v1/monitoring/drift").json()
    names = {c["name"] for c in report["checks"]}
    assert {"similarity_psi", "category_psi", "text_length_psi", "low_confidence_rate_delta",
            "rejection_rate", "embedding_centroid_cosine"} <= names
    assert report["window"]["match_records"] == 10
    assert report["reference"]["n"] == 6 and report["reference"]["source"] == "computed"
    assert report["status"] in {"warning", "alert"}


def test_drift_uses_exported_profile(artifacts, settings, encoder, profile):
    (artifacts / "reference_profile.json").write_text(replace(profile, model="hashing").to_json())
    # The profile only counts if the cohort embeddings were loaded from disk; with
    # synthetic data they are encoded at startup, so the service rebuilds it.
    with TestClient(create_app(settings=settings, encoder=encoder)) as c:
        assert c.get("/v1/monitoring/drift").json()["reference"]["source"] == "computed"


def test_metrics_endpoint(client):
    client.post("/v1/match", json={"text": ACCOUNTANT_RESUME})
    client.get("/no/such/path")
    text = client.get("/metrics").text
    assert 'screening_http_requests_total{method="POST",route="/v1/match",status="200"} 1' in text
    assert 'route="unmatched"' in text                       # raw 404 paths are not labels
    assert "/no/such/path" not in text
    assert 'screening_predictions_total{endpoint="match",status="ok"} 1' in text
    assert "screening_http_request_duration_seconds_bucket" in text
    assert "screening_model_ready 1" in text
    assert 'screening_drift_status{check="similarity_psi"}' in text


def test_metrics_can_be_disabled(settings, encoder):
    with TestClient(create_app(settings=replace(settings, metrics_enabled=False), encoder=encoder)) as c:
        assert c.get("/metrics").status_code == 404


def test_monitoring_requires_api_key_when_set(settings, encoder):
    with TestClient(create_app(settings=replace(settings, api_key="s3cret"), encoder=encoder)) as c:
        assert c.get("/v1/monitoring/drift").status_code == 401
        assert c.get("/v1/monitoring/drift", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.get("/metrics").status_code == 200  # scrape endpoint stays open, like the probes


def test_model_info_reports_build_and_monitoring(client):
    info = client.get("/v1/model").json()
    assert info["api_version"] == "1.1.0"
    assert info["monitoring"]["drift_reference"]["n"] == 6
    assert info["build"]


def test_json_log_format(settings, encoder, capfd):
    with TestClient(create_app(settings=replace(settings, log_format="json", log_predictions_stdout=True),
                               encoder=encoder)) as c:
        c.post("/v1/match", json={"text": CHEF_RESUME})
    lines = [ln for ln in capfd.readouterr().err.splitlines() if ln.startswith("{")]
    parsed = [json.loads(ln) for ln in lines]
    preds = [p for p in parsed if p.get("logger") == "screening.predictions"]
    assert preds and preds[0]["prediction"]["best_category"] == "CHEF"
    assert any(p.get("logger") == "screening.api" for p in parsed)
