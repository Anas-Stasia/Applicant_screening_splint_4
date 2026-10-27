"""FastAPI application — the HTTP layer around ``MatchingService``.

Run it (from the ``serving/`` folder):

    uvicorn app.main:app --host 0.0.0.0 --port 8000

then open http://localhost:8000/docs for interactive, auto-generated API docs.

Architecture
------------
    client (Frontend / Backend squad)
        │  JSON over HTTP
        ▼
    middleware      request id, timing header, CORS, gzip, Prometheus metrics
        ▼
    route           Pydantic validates the payload (schemas.py)  → 422 on bad shape
        ▼
    service         clean → validate → cache lookup → batch encode → cosine → top-k
        ▼
    response        Pydantic serialises the result; errors use one envelope (errors.py)
        ▼
    monitoring      every scored item → prediction log (JSONL) → drift report / metrics

Endpoints are plain ``def`` (not ``async def``) on purpose: model inference is
CPU-bound, so FastAPI runs them in its threadpool and the event loop stays free
to accept other connections while a batch is being encoded.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Depends, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .encoders import Encoder, build_encoder
from .errors import AppError, InputRejected, ModelNotReady, Unauthorized, register_error_handlers
from .monitoring import DriftMonitor, Metrics, PredictionLog
from .schemas import (
    BatchMatchRequest,
    BatchMatchResponse,
    BatchSummary,
    CategoriesResponse,
    CohortRankRequest,
    CohortRankResponse,
    DriftReport,
    ErrorResponse,
    HealthResponse,
    MatchRequest,
    MatchResponse,
    MatchResult,
    ModelInfoResponse,
    PredictionsResponse,
    ReadyResponse,
    ScreenRequest,
    ScreenResponse,
)
from .service import MatchingService
from .settings import Settings, get_settings
from .text import clean_text, text_key

logger = logging.getLogger("screening.api")

API_VERSION = "1.1.0"  # 1.1: Sprint 4 monitoring endpoints (additive, no breaking change)


class JsonFormatter(logging.Formatter):
    """One JSON object per line — what Render, Railway, CloudWatch and Loki parse best."""

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if record.name == "screening.predictions":  # already JSON: embed, don't double-encode
            try:
                return json.dumps({"ts": self.formatTime(record), "level": record.levelname,
                                   "logger": record.name, "prediction": json.loads(msg)})
            except ValueError:
                pass
        out = {"ts": self.formatTime(record), "level": record.levelname, "logger": record.name, "message": msg}
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out)


def configure_logging(fmt: str = "text") -> None:
    # Configure only this app's loggers (not the root logger), so we don't turn on
    # INFO logging for every library in the process (httpx, torch, ...).
    root = logging.getLogger("screening")
    for h in list(root.handlers):
        if getattr(h, "_screening", False):
            root.removeHandler(h)
    handler = logging.StreamHandler()
    handler._screening = True  # type: ignore[attr-defined]
    handler.setFormatter(JsonFormatter() if fmt == "json"
                         else logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False

_ERRORS = {
    401: {"model": ErrorResponse, "description": "Missing or wrong API key (only when SCREENING_API_KEY is set)."},
    413: {"model": ErrorResponse, "description": "Batch larger than the configured limit."},
    422: {"model": ErrorResponse, "description": "Invalid payload, or text that cannot be scored."},
    503: {"model": ErrorResponse, "description": "Model still loading."},
}


def create_app(settings: Settings | None = None, encoder: Encoder | None = None) -> FastAPI:
    """App factory. Tests inject a lightweight encoder; production builds the real one."""
    settings = settings or get_settings()
    configure_logging(settings.log_format)

    # Sprint 4: monitoring objects live for the whole process (not per request).
    predictions = PredictionLog(settings.log_dir, window=settings.drift_window,
                                to_stdout=settings.log_predictions_stdout)
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        enc = encoder or build_encoder(settings.model_name, settings.device)
        service = MatchingService(settings, enc)
        service.embedding_observer = predictions.observe_embeddings
        service.load()  # model + embeddings loaded ONCE, before traffic is accepted
        app.state.service = service
        app.state.drift = DriftMonitor(service.reference, settings.drift_min_samples) if service.reference else None
        logger.info("Monitoring: prediction log %s, drift reference %s, env=%s build=%s",
                    predictions.log_dir or "memory-only",
                    service.reference.source if service.reference else "disabled",
                    settings.environment, settings.git_sha)
        yield
        app.state.service = None

    app = FastAPI(
        title="Applicant Screening API",
        version=API_VERSION,
        description=(
            "Scores resumes against job categories and job descriptions using "
            "Sentence-Transformers embeddings (Sprint 1) with the guardrails from Sprint 2. "
            "All errors share the envelope `{\"error\": {code, message, details, request_id}}`."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.predictions = predictions
    app.state.metrics = metrics
    app.state.drift = None

    # --- middleware -------------------------------------------------------
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        # Honour an incoming X-Request-ID so one id can be traced across the
        # Backend squad's service and this one.
        request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        # Label by route *template* (/v1/match), never the raw URL, so a scanner
        # hitting random paths can't blow up the number of metric series.
        route = request.scope.get("route")
        metrics.observe_request(request.method, getattr(route, "path", "unmatched"),
                                response.status_code, elapsed_ms / 1000)
        logger.info("%s %s -> %d in %.1fms [%s]", request.method, request.url.path,
                    response.status_code, elapsed_ms, request.state.request_id)
        return response

    register_error_handlers(app)

    # --- dependencies -----------------------------------------------------
    def get_service(request: Request) -> MatchingService:
        service = getattr(request.app.state, "service", None)
        if service is None or not service.ready:
            raise ModelNotReady("The model is still loading. Retry shortly.")
        return service

    def check_api_key(request: Request) -> None:
        if settings.api_key and request.headers.get("X-API-Key") != settings.api_key:
            raise Unauthorized("Missing or invalid X-API-Key header.")

    v1 = [Depends(check_api_key)]

    def _ms(t0: float) -> float:
        return round((time.perf_counter() - t0) * 1000, 2)

    # --- prediction logging (Sprint 4) ---------------------------------------
    def _base(request: Request, endpoint: str, latency_ms: float, model: str) -> dict:
        return {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "request_id": getattr(request.state, "request_id", None), "endpoint": endpoint,
                "model": model, "api_version": API_VERSION, "build": settings.git_sha,
                "env": settings.environment, "latency_ms": latency_ms}

    def _text_fields(raw: str | None) -> dict:
        cleaned = clean_text(raw or "")
        # 16 hex chars of SHA-256: enough to spot the same resume being re-sent,
        # useless for recovering its content. The text itself is never logged.
        return {"text_chars": len(cleaned), "text_sha": text_key(cleaned)[:16] if cleaned else None}

    def _reject_code(warnings: list[str]) -> str | None:
        return warnings[0].split(":", 1)[0] if warnings else None

    def log_matches(request, endpoint, items, results, latency_ms, model):
        recs = []
        for item, r in zip(items, results):
            rec = _base(request, endpoint, latency_ms, model) | _text_fields(item.text) | {
                "item_id": r.id, "status": r.status, "best_category": r.best_category,
                "similarity": r.similarity, "cached": r.cached,
                "top_categories": [m.category for m in r.top_matches],
            }
            if r.status == "rejected":
                rec["reject_code"] = _reject_code(r.warnings)
            recs.append(rec)
        _emit(recs)

    def log_screen(request, body: ScreenRequest, label, ranked, rejected, latency_ms, model):
        text_by_id = {(c.id if c.id is not None else str(i)): c.text for i, c in enumerate(body.candidates)}
        recs = []
        for c in [*ranked, *rejected]:
            rec = _base(request, "screen", latency_ms, model) | _text_fields(text_by_id.get(c.id)) | {
                "item_id": c.id, "status": c.status, "job": label, "rank": c.rank, "similarity": c.similarity,
            }
            if c.status == "rejected":
                rec["reject_code"] = _reject_code(c.warnings)
            recs.append(rec)
        _emit(recs)

    def _emit(recs: list[dict]) -> None:
        try:
            predictions.record(recs)
            metrics.observe_predictions(recs)
        except Exception:  # logging must never turn a good prediction into a 500
            logger.exception("prediction logging failed")

    # --- ops endpoints ----------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["ops"], summary="Liveness probe")
    def health():
        return HealthResponse()

    @app.get("/ready", response_model=ReadyResponse, tags=["ops"], summary="Readiness probe",
             responses={503: _ERRORS[503]})
    def ready(service: MatchingService = Depends(get_service)):
        return ReadyResponse(ready=True, model=service.encoder.name)

    # --- v1 API -----------------------------------------------------------
    @app.get("/v1/categories", response_model=CategoriesResponse, tags=["reference"],
             dependencies=v1, summary="List job categories the model can match against")
    def categories(service: MatchingService = Depends(get_service)):
        return CategoriesResponse(categories=service.categories, count=len(service.categories))

    @app.get("/v1/model", response_model=ModelInfoResponse, tags=["reference"], dependencies=v1,
             summary="Model metadata, limits and cache statistics")
    def model_info(service: MatchingService = Depends(get_service)):
        s = service.settings
        return ModelInfoResponse(
            model=service.encoder.name,
            embedding_dim=service.encoder.dim,
            max_seq_length=service.encoder.max_seq_length,
            categories=len(service.categories),
            cohort_size=len(service.cohort_ids),
            similarity_floor=s.similarity_floor,
            limits={"max_batch_size": s.max_batch_size, "min_text_chars": s.min_text_chars,
                    "max_text_chars": s.max_text_chars, "max_top_k": s.max_top_k},
            cache=service.cache.stats(),
            job_embeddings_source=service.job_source,
            cohort_embeddings_source=service.cohort_source,
            api_version=API_VERSION,
            build=settings.git_sha,
            environment=settings.environment,
            monitoring={"prediction_log_dir": str(predictions.log_dir) if predictions.log_dir else None,
                        "log_predictions_stdout": settings.log_predictions_stdout,
                        "drift_window": settings.drift_window,
                        "drift_min_samples": settings.drift_min_samples,
                        "drift_reference": service.reference.summary() if service.reference else None},
        )

    @app.post("/v1/match", response_model=MatchResponse, tags=["inference"], dependencies=v1,
              responses={k: _ERRORS[k] for k in (401, 422, 503)},
              summary="Classify one resume into its best-matching job categories")
    def match(body: MatchRequest, request: Request, service: MatchingService = Depends(get_service)):
        t0 = time.perf_counter()
        try:
            result = service.match_one(body.text, body.top_k)
        except AppError as exc:
            if isinstance(exc, InputRejected):  # unscoreable text still counts toward the rejection rate
                rej = MatchResult(id="0", status="rejected", warnings=[f"{exc.code}: {exc.message}"])
                log_matches(request, "match", [body], [rej], _ms(t0), service.encoder.name)
            raise
        latency = _ms(t0)
        log_matches(request, "match", [body], [result], latency, service.encoder.name)
        return MatchResponse(result=result, model=service.encoder.name, latency_ms=latency)

    @app.post("/v1/match/batch", response_model=BatchMatchResponse, tags=["inference"], dependencies=v1,
              responses=_ERRORS,
              summary="Classify many resumes in one call (partial failures are reported per item)")
    def match_batch(body: BatchMatchRequest, request: Request, service: MatchingService = Depends(get_service)):
        t0 = time.perf_counter()
        results = service.match(body.items, body.top_k)
        log_matches(request, "match_batch", body.items, results, _ms(t0), service.encoder.name)
        summary = BatchSummary(
            total=len(results),
            ok=sum(r.status == "ok" for r in results),
            low_confidence=sum(r.status == "low_confidence" for r in results),
            rejected=sum(r.status == "rejected" for r in results),
        )
        return BatchMatchResponse(results=results, summary=summary,
                                  model=service.encoder.name, latency_ms=_ms(t0))

    @app.post("/v1/screen", response_model=ScreenResponse, tags=["inference"], dependencies=v1,
              responses=_ERRORS,
              summary="Rank a list of applicants against one job (category or free-text description)")
    def screen(body: ScreenRequest, request: Request, service: MatchingService = Depends(get_service)):
        t0 = time.perf_counter()
        label, ranked, rejected = service.screen(body.job, body.candidates, body.top_k)
        latency = _ms(t0)
        log_screen(request, body, label, ranked, rejected, latency, service.encoder.name)
        return ScreenResponse(job=label, ranked=ranked, rejected=rejected,
                              model=service.encoder.name, latency_ms=latency)

    @app.post("/v1/cohort/rank", response_model=CohortRankResponse, tags=["inference"], dependencies=v1,
              responses={k: _ERRORS[k] for k in (401, 422, 503)},
              summary="Rank the indexed Kaggle resume cohort against a job")
    def cohort_rank(body: CohortRankRequest, service: MatchingService = Depends(get_service)):
        t0 = time.perf_counter()
        label, matches = service.rank_cohort(body.job, body.top_k)
        return CohortRankResponse(job=label, matches=matches, cohort_size=len(service.cohort_ids),
                                  model=service.encoder.name, latency_ms=_ms(t0))

    # --- monitoring endpoints (Sprint 4) ------------------------------------------
    @app.get("/v1/monitoring/drift", response_model=DriftReport, tags=["monitoring"], dependencies=v1,
             responses={k: _ERRORS[k] for k in (401, 503)},
             summary="Compare recent predictions with the training cohort (PSI and rate checks)")
    def drift(request: Request, service: MatchingService = Depends(get_service)):
        monitor: DriftMonitor | None = request.app.state.drift
        if monitor is None:
            raise ModelNotReady("Drift monitoring is unavailable: no reference cohort was loaded.")
        return monitor.report(predictions.recent(), predictions.recent_vectors())

    @app.get("/v1/monitoring/predictions", response_model=PredictionsResponse, tags=["monitoring"],
             dependencies=v1, summary="Most recent logged predictions (no resume text)")
    def recent_predictions(limit: int = Query(50, ge=1, le=1000),
                           endpoint: str | None = Query(None, description="match | match_batch | screen")):
        rows = predictions.recent(limit=limit, endpoint=endpoint)
        f = predictions.current_file()
        return PredictionsResponse(records=rows, count=len(rows), total_logged=predictions.total,
                                   window=predictions.window, log_file=f.name if f else None)

    if settings.metrics_enabled:
        @app.get("/metrics", response_class=PlainTextResponse, tags=["ops"], include_in_schema=False)
        def prometheus_metrics(request: Request):
            gauges: dict[str, list] = {}
            service = getattr(request.app.state, "service", None)
            if service is not None:
                cache = service.cache.stats()
                gauges["screening_embedding_cache_entries"] = [({}, cache["size"])]
                gauges["screening_embedding_cache_hits_total"] = [({}, cache["hits"])]
                gauges["screening_embedding_cache_misses_total"] = [({}, cache["misses"])]
            gauges["screening_model_ready"] = [({}, int(bool(service and service.ready)))]
            gauges["screening_build_info"] = [({"version": API_VERSION, "build": settings.git_sha,
                                                "env": settings.environment}, 1)]
            monitor = request.app.state.drift
            if monitor is not None:
                rep_ = monitor.report(predictions.recent(), predictions.recent_vectors())
                level = {"ok": 0, "warning": 1, "alert": 2, "insufficient_data": -1}
                gauges["screening_drift_value"] = [({"check": c["name"]}, c["value"])
                                                   for c in rep_["checks"] if c["value"] is not None]
                gauges["screening_drift_status"] = [({"check": c["name"]}, level[c["status"]])
                                                    for c in rep_["checks"]]
            return metrics.render(gauges)

    # --- demo page ----------------------------------------------------------
    # A minimal same-origin UI (serving/demo/index.html) the Frontend squad can
    # read as a working reference for calling the API from a browser.
    demo_dir = Path(__file__).resolve().parents[1] / "demo"
    if demo_dir.is_dir():
        app.mount("/demo", StaticFiles(directory=demo_dir, html=True), name="demo")

    return app


app = create_app()
