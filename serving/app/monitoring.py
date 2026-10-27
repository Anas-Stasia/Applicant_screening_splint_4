"""Sprint 4 — prediction logging, data-drift monitoring and Prometheus metrics.

Three pieces, each usable on its own:

* ``PredictionLog``  — one JSON line per scored (or rejected) resume, written to a
  daily file and kept in a bounded in-memory window. **No resume text is logged**:
  only a short hash (to spot repeats), the length, and the model's output.
* ``ReferenceProfile`` + ``DriftMonitor`` — what "normal" traffic looks like, taken
  from the 2,481-resume training cohort, and a comparison of recent traffic against
  it using the Population Stability Index (PSI) and a few simple rates.
* ``Metrics`` — request/prediction counters and a latency histogram rendered in the
  Prometheus text format at ``GET /metrics``. No extra dependency.

Why these drift signals?
------------------------
We have no ground-truth labels in production (nobody tells the API who got hired),
so accuracy can't be measured live. Instead we watch the *inputs* and the *model's
confidence*, which move first when the world changes:

=====================  ===================================================  =============
signal                 what a change usually means                          reference
=====================  ===================================================  =============
best-similarity PSI    resumes look less like any job we know               cohort
category-mix PSI       a new hiring campaign, or the model mis-routing      cohort
text-length PSI        a new upload path (PDF parser, form field) changed   cohort
                       what text reaches the API
low-confidence rate    more out-of-scope resumes (the saxophonist case)     cohort (~17%)
rejection rate         a client is sending empty/garbled text               0% (cohort)
embedding centroid     the average resume "meaning" has moved               cohort
=====================  ===================================================  =============

PSI rule of thumb (industry standard): < 0.10 stable, 0.10–0.25 moderate shift,
> 0.25 significant shift. Those cut-offs assume thousands of samples. On a small
window, sampling noise alone produces a PSI of roughly (bins − 1) / n — with 24
categories and 100 resumes that is already ~0.23, so the fixed cut-offs would fire
on perfectly normal traffic. ``psi_thresholds`` therefore raises them to the PSI
that pure noise would exceed only 1% (warn) / 0.1% (alert) of the time. It finds
that level by simulation: draw ``n`` items from the reference distribution 2,000
times and take the 99th / 99.9th percentile of the PSI values (cached per window size).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger("screening.monitoring")
prediction_logger = logging.getLogger("screening.predictions")

PSI_WARN = 0.10
PSI_ALERT = 0.25
_EPS = 1e-4  # smoothing so an empty bin doesn't make PSI infinite

# Endpoints whose similarity is "resume vs. its best category" — the same quantity
# the reference profile is built from. /v1/screen scores against one chosen job,
# so its similarities have a different distribution and are not compared.
MATCH_ENDPOINTS = frozenset({"match", "match_batch"})


# --------------------------------------------------------------------------- PSI
def psi(expected: Iterable[float], actual: Iterable[float]) -> float:
    """Population Stability Index between two discrete distributions (same bins).

    PSI = Σ (a_i − e_i) · ln(a_i / e_i). Both inputs are normalised here, so raw
    counts work as well as proportions.
    """
    e = np.asarray(list(expected), dtype=float)
    a = np.asarray(list(actual), dtype=float)
    e = e / e.sum() if e.sum() > 0 else np.full_like(e, 1 / len(e))
    a = a / a.sum() if a.sum() > 0 else np.full_like(a, 1 / len(a))
    e = np.clip(e, _EPS, None)
    a = np.clip(a, _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


@lru_cache(maxsize=256)
def _noise_psi(probs: tuple[float, ...], n: int, draws: int = 2000) -> tuple[float, float]:
    """99th and 99.9th percentile of PSI between the reference and samples drawn *from* it."""
    rng = np.random.default_rng(12345)  # fixed seed: thresholds are deterministic
    e = np.asarray(probs, dtype=float)
    e = e / e.sum()
    counts = rng.multinomial(n, e, size=draws) / n
    ec, ac = np.clip(e, _EPS, None), np.clip(counts, _EPS, None)
    vals = np.sum((ac - ec) * np.log(ac / ec), axis=1)
    return float(np.quantile(vals, 0.99)), float(np.quantile(vals, 0.999))


def psi_thresholds(probs: Iterable[float], n: int) -> tuple[float, float]:
    """(warn, alert) PSI cut-offs: the fixed rule of thumb, or the noise level if higher."""
    if n <= 0:
        return PSI_WARN, PSI_ALERT
    q99, q999 = _noise_psi(tuple(round(float(x), 6) for x in probs), int(n))
    return round(max(PSI_WARN, q99), 4), round(max(PSI_ALERT, q999), 4)


def _histogram(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Counts per bin, with open-ended first/last bins so nothing falls outside."""
    inner = np.asarray(edges[1:-1], dtype=float)
    idx = np.searchsorted(inner, values, side="right")
    return np.bincount(idx, minlength=len(edges) - 1).astype(float)


def _quantile_edges(values: np.ndarray, n_bins: int) -> list[float]:
    qs = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(np.round(qs, 6)).tolist()
    edges[0], edges[-1] = float("-inf"), float("inf")
    return edges


def _json_safe_edges(edges: list[float]) -> list[float | str]:
    return ["-inf" if e == float("-inf") else "inf" if e == float("inf") else e for e in edges]


def _parse_edges(edges: list) -> list[float]:
    return [float(e) for e in edges]  # float("inf") / float("-inf") parse the strings back


# ------------------------------------------------------------- reference profile
@dataclass
class ReferenceProfile:
    """Snapshot of what the training cohort looks like to the deployed model."""

    model: str
    n: int
    similarity_floor: float
    similarity_edges: list[float]
    similarity_probs: list[float]
    length_edges: list[float]
    length_probs: list[float]
    category_probs: dict[str, float]
    low_confidence_rate: float
    centroid: list[float]
    centroid_floor: float  # lowest centroid cosine seen for random cohort samples
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    source: str = "computed"

    @classmethod
    def build(
        cls,
        *,
        model: str,
        categories: list[str],
        job_embeddings: np.ndarray,
        cohort_embeddings: np.ndarray,
        text_lengths: np.ndarray,
        similarity_floor: float,
        n_bins: int = 10,
        sample_size: int = 100,
        seed: int = 42,
    ) -> ReferenceProfile:
        sims = cohort_embeddings @ job_embeddings.T
        best = sims.max(axis=1)
        pred = np.asarray(categories)[sims.argmax(axis=1)]

        sim_edges = _quantile_edges(best, n_bins)
        len_edges = _quantile_edges(np.asarray(text_lengths, dtype=float), n_bins)
        counts = Counter(pred.tolist())
        cat_probs = {c: counts.get(c, 0) / len(pred) for c in categories}

        centroid = cohort_embeddings.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) or 1.0)
        # How far does the centroid of a *normal* random sample wander? Use the worst
        # of 200 draws as the floor: below it, the traffic really is different.
        rng = np.random.default_rng(seed)
        k = min(sample_size, len(cohort_embeddings))
        cosines = []
        for _ in range(200):
            m = cohort_embeddings[rng.choice(len(cohort_embeddings), k, replace=False)].mean(axis=0)
            cosines.append(float(m @ centroid / (np.linalg.norm(m) or 1.0)))

        return cls(
            model=model,
            n=int(len(best)),
            similarity_floor=similarity_floor,
            similarity_edges=sim_edges,
            similarity_probs=(_histogram(best, sim_edges) / len(best)).round(6).tolist(),
            length_edges=len_edges,
            length_probs=(_histogram(np.asarray(text_lengths, float), len_edges) / len(text_lengths)).round(6).tolist(),
            category_probs={c: round(p, 6) for c, p in cat_probs.items()},
            low_confidence_rate=round(float((best < similarity_floor).mean()), 6),
            centroid=centroid.round(6).tolist(),
            centroid_floor=round(min(cosines), 6),
        )

    # JSON round-trip (±inf edges are stored as strings, since JSON has no infinity).
    def to_json(self) -> str:
        d = asdict(self)
        d["similarity_edges"] = _json_safe_edges(self.similarity_edges)
        d["length_edges"] = _json_safe_edges(self.length_edges)
        return json.dumps(d, indent=1)

    @classmethod
    def from_json(cls, text: str) -> ReferenceProfile:
        d = json.loads(text)
        d["similarity_edges"] = _parse_edges(d["similarity_edges"])
        d["length_edges"] = _parse_edges(d["length_edges"])
        d["source"] = "loaded"
        return cls(**d)

    def summary(self) -> dict:
        return {"model": self.model, "n": self.n, "created_at": self.created_at, "source": self.source,
                "low_confidence_rate": self.low_confidence_rate, "similarity_floor": self.similarity_floor}


# --------------------------------------------------------------- prediction log
class PredictionLog:
    """Append-only JSONL prediction log + a bounded in-memory window for drift checks.

    Files are named ``predictions-YYYY-MM-DD.jsonl`` (UTC) so they rotate daily and
    can be shipped to S3 / BigQuery / a notebook as-is. Writes are a single
    ``write()`` of one line under a lock, so concurrent threads never interleave.
    """

    def __init__(self, log_dir: Path | None, window: int = 1000, to_stdout: bool = False) -> None:
        self.log_dir = Path(log_dir) if log_dir else None
        self.to_stdout = to_stdout
        self._records: deque[dict] = deque(maxlen=window)
        self._vectors: deque[np.ndarray] = deque(maxlen=window)
        self._lock = threading.Lock()
        self.total = 0
        self.write_errors = 0
        if self.log_dir:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:  # read-only filesystem etc. — keep serving
                logger.warning("Prediction log dir %s is not writable (%s); file logging disabled.",
                               self.log_dir, exc)
                self.log_dir = None

    @property
    def window(self) -> int:
        return self._records.maxlen or 0

    def current_file(self) -> Path | None:
        if not self.log_dir:
            return None
        return self.log_dir / f"predictions-{datetime.now(UTC):%Y-%m-%d}.jsonl"

    def record(self, records: list[dict]) -> None:
        if not records:
            return
        lines = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records)
        with self._lock:
            self._records.extend(records)
            self.total += len(records)
            path = self.current_file()
            if path is not None:
                try:
                    with path.open("a", encoding="utf-8") as fh:
                        fh.write(lines)
                except OSError:
                    self.write_errors += 1
                    logger.exception("Could not write prediction log %s", path)
        if self.to_stdout:
            for r in records:
                prediction_logger.info(json.dumps(r, separators=(",", ":")))

    def observe_embeddings(self, matrix: np.ndarray) -> None:
        with self._lock:
            self._vectors.extend(np.asarray(matrix, dtype=np.float32))

    def recent(self, limit: int | None = None, endpoint: str | None = None) -> list[dict]:
        with self._lock:
            rows = list(self._records)
        if endpoint:
            rows = [r for r in rows if r.get("endpoint") == endpoint]
        return rows[-limit:] if limit else rows

    def recent_vectors(self) -> np.ndarray | None:
        with self._lock:
            return np.vstack(self._vectors) if self._vectors else None

    @staticmethod
    def read_files(paths: Iterable[Path]) -> list[dict]:
        """Load JSONL files written by this class (used by scripts/drift_report.py)."""
        rows: list[dict] = []
        for p in paths:
            with Path(p).open(encoding="utf-8") as fh:
                rows.extend(json.loads(line) for line in fh if line.strip())
        return rows


# --------------------------------------------------------------- drift monitor
def _status_for(value: float, warn: float, alert: float, higher_is_worse: bool = True) -> str:
    if not higher_is_worse:
        value, warn, alert = -value, -warn, -alert
    return "alert" if value >= alert else "warning" if value >= warn else "ok"


class DriftMonitor:
    """Compares a set of logged predictions with the reference profile."""

    def __init__(self, profile: ReferenceProfile, min_samples: int = 100) -> None:
        self.profile = profile
        self.min_samples = min_samples

    def report(self, records: list[dict], vectors: np.ndarray | None = None) -> dict:
        p = self.profile
        scored = [r for r in records if r.get("status") != "rejected"]
        matched = [r for r in scored if r.get("endpoint") in MATCH_ENDPOINTS and r.get("similarity") is not None]
        checks: list[dict] = []

        def add(name, value, warn, alert, n, detail, higher_is_worse=True):
            enough = n >= self.min_samples
            checks.append({
                "name": name,
                "value": None if value is None else round(float(value), 4),
                "warn_at": warn, "alert_at": alert, "samples": n,
                "status": _status_for(value, warn, alert, higher_is_worse) if enough and value is not None
                else "insufficient_data",
                "detail": detail,
            })

        # 1. Model confidence: best-category similarity distribution
        if matched:
            sims = np.array([r["similarity"] for r in matched], dtype=float)
            v = psi(p.similarity_probs, _histogram(sims, p.similarity_edges))
            add("similarity_psi", v, *psi_thresholds(p.similarity_probs, len(matched)), len(matched),
                f"median best similarity {np.median(sims):.3f} (reference deciles from {p.n} cohort resumes)")
            low = float((sims < p.similarity_floor).mean())
            add("low_confidence_rate_delta", low - p.low_confidence_rate, 0.10, 0.20, len(matched),
                f"{low:.1%} below the {p.similarity_floor} floor vs {p.low_confidence_rate:.1%} in the cohort")

            # 2. Prediction mix: which categories the model is choosing
            cats = list(p.category_probs)
            counts = Counter(r.get("best_category") for r in matched)
            v = psi([p.category_probs[c] for c in cats], [counts.get(c, 0) for c in cats])
            top = ", ".join(f"{c} {n / len(matched):.0%}" for c, n in counts.most_common(3))
            add("category_psi", v, *psi_thresholds([p.category_probs[c] for c in cats], len(matched)), len(matched),
                f"most predicted: {top}")
        else:
            for name in ("similarity_psi", "low_confidence_rate_delta", "category_psi"):
                add(name, None, 0, 0, 0, "no /v1/match traffic in the window yet")

        # 3. Input shape: text length of everything that was scored (any endpoint)
        if scored:
            lengths = np.array([r.get("text_chars", 0) for r in scored], dtype=float)
            v = psi(p.length_probs, _histogram(lengths, p.length_edges))
            add("text_length_psi", v, *psi_thresholds(p.length_probs, len(scored)), len(scored),
                f"median length {np.median(lengths):,.0f} chars")
        else:
            add("text_length_psi", None, 0, 0, 0, "no scored traffic yet")

        # 4. Data quality: how much traffic can't be scored at all
        if records:
            rej = sum(r.get("status") == "rejected" for r in records) / len(records)
            add("rejection_rate", rej, 0.05, 0.15, len(records), f"{rej:.1%} of items rejected (cohort: 0%)")
        else:
            add("rejection_rate", None, 0, 0, 0, "no traffic yet")

        # 5. Semantic shift: cosine between recent mean embedding and the cohort's
        if vectors is not None and len(vectors):
            m = vectors.mean(axis=0)
            cos = float(m @ np.asarray(p.centroid) / (np.linalg.norm(m) or 1.0))
            floor = p.centroid_floor
            add("embedding_centroid_cosine", cos, round(floor, 4), round(floor - 0.05, 4), len(vectors),
                f"cosine to cohort centroid; random cohort samples never went below {floor:.3f}",
                higher_is_worse=False)
        else:
            add("embedding_centroid_cosine", None, 0, 0, 0, "no embeddings observed yet")

        statuses = {c["status"] for c in checks}
        overall = ("alert" if "alert" in statuses else "warning" if "warning" in statuses
                   else "ok" if "ok" in statuses else "insufficient_data")
        ts = [r["ts"] for r in records if "ts" in r]
        return {
            "status": overall,
            "window": {"records": len(records), "scored": len(scored), "match_records": len(matched),
                       "from": min(ts) if ts else None, "to": max(ts) if ts else None,
                       "min_samples": self.min_samples},
            "checks": checks,
            "reference": p.summary(),
            "thresholds": {"psi_warn": PSI_WARN, "psi_alert": PSI_ALERT,
                           "note": "PSI cut-offs are raised above these on small windows (see warn_at/alert_at)"},
        }


# ------------------------------------------------------------------- metrics
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class Metrics:
    """Tiny thread-safe Prometheus exporter (counters + one histogram)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started = time.time()
        self.requests: Counter = Counter()        # (method, route, status) -> n
        self.predictions: Counter = Counter()     # (endpoint, status) -> n
        self._lat_buckets: dict[str, list[int]] = {}
        self._lat_sum: Counter = Counter()
        self._lat_count: Counter = Counter()

    def observe_request(self, method: str, route: str, status: int, seconds: float) -> None:
        with self._lock:
            self.requests[(method, route, str(status))] += 1
            b = self._lat_buckets.setdefault(route, [0] * len(_LATENCY_BUCKETS))
            for i, edge in enumerate(_LATENCY_BUCKETS):
                if seconds <= edge:
                    b[i] += 1
            self._lat_sum[route] += seconds
            self._lat_count[route] += 1

    def observe_predictions(self, records: list[dict]) -> None:
        with self._lock:
            for r in records:
                self.predictions[(r["endpoint"], r["status"])] += 1

    def render(self, extra_gauges: dict[str, list[tuple[dict, float]]] | None = None) -> str:
        def labels(d: dict) -> str:
            return "{" + ",".join(f'{k}="{v}"' for k, v in d.items()) + "}" if d else ""

        out: list[str] = []
        with self._lock:
            out += ["# HELP screening_uptime_seconds Seconds since the process started.",
                    "# TYPE screening_uptime_seconds gauge",
                    f"screening_uptime_seconds {time.time() - self.started:.1f}"]
            out += ["# HELP screening_http_requests_total HTTP requests by route and status.",
                    "# TYPE screening_http_requests_total counter"]
            for (m, r, s), n in sorted(self.requests.items()):
                out.append(f"screening_http_requests_total{labels({'method': m, 'route': r, 'status': s})} {n}")
            out += ["# HELP screening_http_request_duration_seconds Request latency.",
                    "# TYPE screening_http_request_duration_seconds histogram"]
            for route, buckets in sorted(self._lat_buckets.items()):
                for edge, n in zip(_LATENCY_BUCKETS, buckets):
                    out.append(f"screening_http_request_duration_seconds_bucket"
                               f"{labels({'route': route, 'le': edge})} {n}")
                out.append(f"screening_http_request_duration_seconds_bucket"
                           f"{labels({'route': route, 'le': '+Inf'})} {self._lat_count[route]}")
                out.append(f"screening_http_request_duration_seconds_sum{labels({'route': route})} "
                           f"{self._lat_sum[route]:.6f}")
                out.append(f"screening_http_request_duration_seconds_count{labels({'route': route})} "
                           f"{self._lat_count[route]}")
            out += ["# HELP screening_predictions_total Scored resumes by endpoint and status.",
                    "# TYPE screening_predictions_total counter"]
            for (e, s), n in sorted(self.predictions.items()):
                out.append(f"screening_predictions_total{labels({'endpoint': e, 'status': s})} {n}")
        for name, series in (extra_gauges or {}).items():
            out += [f"# TYPE {name} gauge"]
            out += [f"{name}{labels(lbl)} {val}" for lbl, val in series]
        return "\n".join(out) + "\n"
