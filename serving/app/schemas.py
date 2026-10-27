"""Request/response payload schemas (Pydantic v2).

These classes are the contract with the Frontend and Backend squads: FastAPI turns
them into the OpenAPI spec served at ``/openapi.json`` and the interactive docs at
``/docs``, and they validate every request before any model code runs.

Shape rules (types, required fields, ranges) live here. Content rules that depend
on runtime settings — "too short to score", "batch too large" — live in the
service, so they return the same error codes from single and batch endpoints.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MatchStatus = Literal["ok", "low_confidence", "rejected"]


class _Strict(BaseModel):
    # Reject unknown fields so client typos ("txt" instead of "text") fail loudly
    # instead of silently being ignored.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
class ResumeIn(_Strict):
    id: str | None = Field(
        default=None,
        max_length=128,
        description="Caller's own identifier (applicant id, upload id...). "
        "Echoed back so results can be joined to your records. "
        "Defaults to the item's position in the batch.",
        examples=["applicant-1042"],
    )
    text: str = Field(
        description="Resume text. Plain text or HTML; it is cleaned server-side.",
        examples=["Senior accountant with 8 years of experience in tax, payroll and ledger reconciliation."],
    )


class JobIn(_Strict):
    """A job to screen against: either a known category or free-text description."""

    category: str | None = Field(
        default=None,
        description="One of the categories from GET /v1/categories (case-insensitive).",
        examples=["ACCOUNTANT"],
    )
    description: str | None = Field(
        default=None,
        description="Free-text job description, e.g. pasted from a real job posting.",
        examples=["We are hiring a payroll accountant familiar with general ledger and tax filings."],
    )

    @model_validator(mode="after")
    def exactly_one(self) -> JobIn:
        if bool(self.category) == bool(self.description):
            raise ValueError("Provide exactly one of 'category' or 'description'.")
        return self


def _check_unique_ids(items: list[ResumeIn]) -> list[ResumeIn]:
    ids = [i.id for i in items if i.id is not None]
    dupes = sorted({x for x in ids if ids.count(x) > 1})
    if dupes:
        raise ValueError(f"Duplicate item ids: {dupes}")
    return items


class MatchRequest(_Strict):
    text: str = Field(description="Resume text to classify.", examples=ResumeIn.model_fields["text"].examples)
    top_k: int = Field(default=3, ge=1, le=24, description="How many categories to return.")


class BatchMatchRequest(_Strict):
    items: list[ResumeIn] = Field(min_length=1, description="Resumes to classify in one call.")
    top_k: int = Field(default=3, ge=1, le=24)

    @field_validator("items")
    @classmethod
    def unique_ids(cls, items: list[ResumeIn]) -> list[ResumeIn]:
        return _check_unique_ids(items)


class ScreenRequest(_Strict):
    job: JobIn
    candidates: list[ResumeIn] = Field(min_length=1, description="Applicants to rank for this job.")
    top_k: int | None = Field(default=None, ge=1, description="Return only the best N. Omit for all.")

    @field_validator("candidates")
    @classmethod
    def unique_ids(cls, items: list[ResumeIn]) -> list[ResumeIn]:
        return _check_unique_ids(items)


class CohortRankRequest(_Strict):
    job: JobIn
    top_k: int = Field(default=10, ge=1, description="How many resumes to return.")


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
class CategoryScore(BaseModel):
    category: str
    similarity: float = Field(description="Cosine similarity in [-1, 1]; higher is a closer match.")


class MatchResult(BaseModel):
    id: str
    status: MatchStatus = Field(
        description="'ok' = confident match; 'low_confidence' = best match is below the "
        "similarity floor (route to manual review); 'rejected' = input could not be scored."
    )
    best_category: str | None = None
    similarity: float | None = None
    top_matches: list[CategoryScore] = []
    warnings: list[str] = []
    cached: bool = Field(default=False, description="True if the embedding came from the cache.")


class MatchResponse(BaseModel):
    result: MatchResult
    model: str
    latency_ms: float


class BatchSummary(BaseModel):
    total: int
    ok: int
    low_confidence: int
    rejected: int


class BatchMatchResponse(BaseModel):
    results: list[MatchResult]
    summary: BatchSummary
    model: str
    latency_ms: float


class RankedCandidate(BaseModel):
    id: str
    rank: int | None = Field(description="1 = best. Null for rejected candidates.")
    similarity: float | None = None
    status: MatchStatus
    warnings: list[str] = []


class ScreenResponse(BaseModel):
    job: str = Field(description="Category name, or 'custom' for a free-text description.")
    ranked: list[RankedCandidate]
    rejected: list[RankedCandidate]
    model: str
    latency_ms: float


class CohortMatch(BaseModel):
    rank: int
    resume_id: str
    category: str
    similarity: float


class CohortRankResponse(BaseModel):
    job: str
    matches: list[CohortMatch]
    cohort_size: int
    model: str
    latency_ms: float


class CategoriesResponse(BaseModel):
    categories: list[str]
    count: int


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    ready: bool
    model: str | None = None


class ModelInfoResponse(BaseModel):
    model: str
    embedding_dim: int
    max_seq_length: int | None
    categories: int
    cohort_size: int
    similarity_floor: float
    limits: dict[str, int]
    cache: dict[str, int]
    job_embeddings_source: str
    cohort_embeddings_source: str
    api_version: str | None = None
    build: str | None = Field(default=None, description="Git commit the running image was built from.")
    environment: str | None = None
    monitoring: dict | None = Field(default=None, description="Prediction-log and drift settings.")


# ---------------------------------------------------------------------------
# Sprint 4: monitoring
# ---------------------------------------------------------------------------
DriftStatus = Literal["ok", "warning", "alert", "insufficient_data"]


class DriftCheck(BaseModel):
    name: str = Field(examples=["similarity_psi"])
    value: float | None
    warn_at: float
    alert_at: float
    samples: int
    status: DriftStatus
    detail: str


class DriftReport(BaseModel):
    status: DriftStatus = Field(description="Worst status across all checks.")
    window: dict
    checks: list[DriftCheck]
    reference: dict
    thresholds: dict


class PredictionRecord(BaseModel):
    """One logged prediction. Resume text is never stored — only a hash and its length."""

    model_config = ConfigDict(extra="allow")
    ts: str
    request_id: str | None
    endpoint: str
    item_id: str
    status: MatchStatus
    text_chars: int | None = None
    text_sha: str | None = None


class PredictionsResponse(BaseModel):
    records: list[PredictionRecord]
    count: int
    total_logged: int
    window: int
    log_file: str | None


class ErrorBody(BaseModel):
    code: str = Field(examples=["TEXT_TOO_SHORT"])
    message: str
    details: list[dict] | dict | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
