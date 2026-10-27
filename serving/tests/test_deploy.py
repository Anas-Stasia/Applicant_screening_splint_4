"""Sprint 4: deployment configuration and documentation stay consistent with the code.

A typo in render.yaml or a renamed endpoint that the docs still advertise doesn't
fail any unit test — it fails at 2 a.m. on deploy day. These checks are cheap.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.main import create_app
from app.settings import Settings

SERVING = Path(__file__).resolve().parents[1]
ROOT = SERVING.parent
yaml = pytest.importorskip("yaml")


def test_dockerfile_production_settings():
    df = (SERVING / "Dockerfile").read_text()
    assert "${PORT}" in df, "cloud hosts inject $PORT; the CMD must use it"
    assert re.search(r"^USER\s+(?!root)\w+", df, re.M), "container must not run as root"
    assert "HEALTHCHECK" in df and "/ready" in df
    assert "COPY serving/artifacts /data" in df
    assert "HF_HUB_OFFLINE=1" in df, "model is baked in; startup must not need the internet"
    for src in re.findall(r"^COPY\s+(\S+)", df, re.M):
        if not src.startswith("--"):
            assert (ROOT / src).exists(), f"Dockerfile copies missing path {src}"


def test_dockerignore_keeps_what_the_image_needs():
    ignored = [ln.strip() for ln in (ROOT / ".dockerignore").read_text().splitlines()
               if ln.strip() and not ln.startswith("#")]
    assert ".git" in ignored and "data_pipeline" in ignored
    assert not any(p in ignored for p in ("serving", "serving/app", "serving/artifacts"))


def test_render_blueprint():
    svc = yaml.safe_load((ROOT / "render.yaml").read_text())["services"][0]
    assert svc["runtime"] == "docker" and svc["healthCheckPath"] == "/ready"
    assert svc["autoDeployTrigger"] == "checksPass", "deploy only commits whose CI passed"
    assert (ROOT / svc["dockerfilePath"]).exists()
    keys = {e["key"] for e in svc["envVars"]}
    assert {"SCREENING_API_KEY", "SCREENING_LOG_PREDICTIONS_STDOUT"} <= keys


def test_railway_config():
    cfg = json.loads((ROOT / "railway.json").read_text())
    assert cfg["build"]["builder"] == "DOCKERFILE"
    assert (ROOT / cfg["build"]["dockerfilePath"]).exists()
    assert cfg["deploy"]["healthcheckPath"] == "/ready"


def test_compose_file():
    api = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]["api"]
    assert api["build"]["dockerfile"] == "serving/Dockerfile"
    assert any(v.endswith(":/srv/logs") for v in api["volumes"]), "prediction logs need a volume"


def test_ci_workflow_references_real_files():
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    jobs = wf["jobs"]
    assert {"lint", "test", "docker"} <= set(jobs)
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    for path in re.findall(r"(serving/[\w/.-]+\.(?:py|txt|toml)|serving/Dockerfile)", text):
        assert (ROOT / path).exists(), f"ci.yml refers to missing {path}"


def test_health_path_exists_in_app(settings, encoder):
    paths = {r.path for r in create_app(settings=settings, encoder=encoder).routes}
    assert {"/health", "/ready", "/metrics", "/v1/monitoring/drift"} <= paths


def test_settings_read_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SCREENING_LOG_DIR", str(tmp_path / "x"))
    monkeypatch.setenv("SCREENING_LOG_PREDICTIONS_STDOUT", "true")
    monkeypatch.setenv("SCREENING_DRIFT_WINDOW", "77")
    monkeypatch.setenv("RENDER_GIT_COMMIT", "abcdef1234567890")
    monkeypatch.delenv("SCREENING_GIT_SHA", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    s = Settings()
    assert s.log_dir == tmp_path / "x" and s.log_predictions_stdout and s.drift_window == 77
    assert s.git_sha == "abcdef123456"
    monkeypatch.setenv("SCREENING_LOG_DIR", "")
    assert Settings().log_dir is None


def test_docs_list_every_endpoint(settings, encoder):
    """Every public route must appear in serving/README.md, so the docs can't fall behind."""
    readme = (SERVING / "README.md").read_text()
    app = create_app(settings=settings, encoder=encoder)
    for r in app.routes:
        path = getattr(r, "path", "")
        if path.startswith("/v1/") or path in {"/health", "/ready", "/metrics"}:
            assert f"`{path}`" in readme, f"{path} is not documented in serving/README.md"
