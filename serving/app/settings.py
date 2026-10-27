"""Runtime configuration for the screening API.

Every value can be overridden with an environment variable (prefix ``SCREENING_``),
so the same code runs on a laptop, in Docker, or behind the backend squad's
gateway without edits. Defaults point at the artifacts Sprint 1/2 already produced
under ``data_pipeline/data/processed``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# serving/app/settings.py -> serving/ -> applicant_screening_nb/
_SERVING_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _SERVING_ROOT.parent
_PIPELINE_ARTIFACTS = _PROJECT_ROOT / "data_pipeline" / "data" / "processed"
# Sprint 4: a slim, committed copy of what the server needs (scripts/export_artifacts.py),
# so a fresh git clone — CI, Docker, Render, Railway — can serve without the notebooks.
_BUNDLED_ARTIFACTS = _SERVING_ROOT / "artifacts"


def _default_artifacts() -> Path:
    if (_PIPELINE_ARTIFACTS / "job_descriptions.csv").exists():
        return _PIPELINE_ARTIFACTS
    return _BUNDLED_ARTIFACTS


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").strip().lower() in {"1", "true", "yes", "on"}


def _git_sha() -> str:
    # Render / Railway set these automatically; CI and Docker builds pass SCREENING_GIT_SHA.
    for var in ("SCREENING_GIT_SHA", "RENDER_GIT_COMMIT", "RAILWAY_GIT_COMMIT_SHA", "GITHUB_SHA"):
        if os.getenv(var):
            return os.environ[var][:12]
    return "dev"


def _env(name: str, default: str) -> str:
    return os.getenv(f"SCREENING_{name}", default)


def _env_list(name: str, default: str) -> list[str]:
    raw = _env(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # --- Model ---
    model_name: str = field(default_factory=lambda: _env("MODEL_NAME", "all-MiniLM-L6-v2"))
    device: str = field(default_factory=lambda: _env("DEVICE", "cpu"))
    encode_batch_size: int = field(default_factory=lambda: int(_env("ENCODE_BATCH_SIZE", "32")))

    # --- Artifacts produced by the Sprint 1/2 notebooks ---
    artifacts_dir: Path = field(
        default_factory=lambda: Path(_env("ARTIFACTS_DIR", str(_default_artifacts())))
    )

    # --- Guardrails carried over from edge_cases.ipynb ---
    min_text_chars: int = field(default_factory=lambda: int(_env("MIN_TEXT_CHARS", "30")))
    max_text_chars: int = field(default_factory=lambda: int(_env("MAX_TEXT_CHARS", "50000")))
    similarity_floor: float = field(default_factory=lambda: float(_env("SIMILARITY_FLOOR", "0.35")))

    # --- Batch / payload limits ---
    max_batch_size: int = field(default_factory=lambda: int(_env("MAX_BATCH_SIZE", "64")))
    max_top_k: int = field(default_factory=lambda: int(_env("MAX_TOP_K", "50")))

    # --- Caching ---
    embedding_cache_size: int = field(default_factory=lambda: int(_env("EMBEDDING_CACHE_SIZE", "4096")))

    # --- Integration ---
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list(
            "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000"
        )
    )
    api_key: str | None = field(default_factory=lambda: os.getenv("SCREENING_API_KEY") or None)

    # --- Sprint 4: logging & monitoring ---
    # Where prediction logs (JSONL, one file per UTC day) go. Empty string = memory only.
    log_dir: Path | None = field(
        default_factory=lambda: Path(v) if (v := _env("LOG_DIR", str(_SERVING_ROOT / "logs"))) else None
    )
    # Also emit each prediction record to stdout — the durable option on hosts with an
    # ephemeral disk (Render, Railway), where stdout is collected by the platform.
    log_predictions_stdout: bool = field(default_factory=lambda: _env_bool("LOG_PREDICTIONS_STDOUT", False))
    log_format: str = field(default_factory=lambda: _env("LOG_FORMAT", "text"))  # "text" | "json"
    drift_window: int = field(default_factory=lambda: int(_env("DRIFT_WINDOW", "1000")))
    drift_min_samples: int = field(default_factory=lambda: int(_env("DRIFT_MIN_SAMPLES", "100")))
    metrics_enabled: bool = field(default_factory=lambda: _env_bool("METRICS_ENABLED", True))
    environment: str = field(default_factory=lambda: _env("ENVIRONMENT", "local"))
    git_sha: str = field(default_factory=_git_sha)

    @property
    def embeddings_dir(self) -> Path:
        return self.artifacts_dir / "embeddings"

    @property
    def job_descriptions_csv(self) -> Path:
        return self.artifacts_dir / "job_descriptions.csv"

    @property
    def resumes_csv(self) -> Path:
        # The committed bundle stores a gzipped copy; pandas reads either transparently.
        plain = self.artifacts_dir / "resumes_clean.csv"
        gz = self.artifacts_dir / "resumes_clean.csv.gz"
        return plain if plain.exists() or not gz.exists() else gz

    @property
    def reference_profile_json(self) -> Path:
        return self.artifacts_dir / "reference_profile.json"


def get_settings() -> Settings:
    return Settings()
