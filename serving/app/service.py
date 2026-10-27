"""Inference service: everything between a validated request and a scored result.

Latency strategy
----------------
* **Load once.** The model, the 24 job-description embeddings and the 2,481 cohort
  resume embeddings are loaded a single time at startup (see ``main.lifespan``),
  never per request.
* **Reuse Sprint 1 work.** Saved ``.npy`` embeddings are loaded instead of
  re-encoded — after a one-row spot check proves they came from the same model.
* **Cache repeats.** Embeddings of incoming text are kept in an LRU cache keyed by
  a hash of the cleaned text, so re-screening the same applicant costs a dict lookup.
* **Batch the misses.** A batch request de-duplicates its texts and sends every
  uncached one to the model in a single ``encode`` call.
* **Cheap maths.** Vectors are L2-normalised, so cosine similarity is one matrix
  product; top-k over the cohort uses ``argpartition`` instead of a full sort.
"""
from __future__ import annotations

import difflib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .encoders import Encoder
from .errors import BatchTooLarge, InferenceFailed, InputRejected, UnknownCategory
from .monitoring import ReferenceProfile
from .schemas import (
    CategoryScore,
    CohortMatch,
    JobIn,
    MatchResult,
    RankedCandidate,
    ResumeIn,
)
from .settings import Settings
from .text import clean_text, letter_count, text_key

logger = logging.getLogger("screening.service")

MIN_LETTERS = 10  # '!!!@@@###$$$' passes a length check but has nothing to embed
SPOT_CHECK_TOLERANCE = 0.99  # cosine between saved and freshly-computed embedding


class EmbeddingCache:
    """Thread-safe LRU cache: text hash -> embedding vector."""

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> np.ndarray | None:
        with self._lock:
            vec = self._data.get(key)
            if vec is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return vec

    def put(self, key: str, vec: np.ndarray) -> None:
        if self.max_size <= 0:
            return
        with self._lock:
            self._data[key] = vec
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"size": len(self._data), "max_size": self.max_size,
                    "hits": self.hits, "misses": self.misses}


@dataclass
class _Prepared:
    """One input after validation: either cleaned text, or the reason it was rejected."""

    id: str
    text: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class MatchingService:
    def __init__(self, settings: Settings, encoder: Encoder) -> None:
        self.settings = settings
        self.encoder = encoder
        self.cache = EmbeddingCache(settings.embedding_cache_size)
        self.ready = False

        self.categories: list[str] = []
        self._category_lookup: dict[str, int] = {}
        self.job_embeddings: np.ndarray | None = None
        self.cohort_ids: list[str] = []
        self.cohort_categories: list[str] = []
        self.cohort_embeddings: np.ndarray | None = None
        self.job_source = "not loaded"
        self.cohort_source = "not loaded"
        # Sprint 4: drift reference + a hook that sees every batch of resume embeddings
        self.reference: ReferenceProfile | None = None
        self.embedding_observer = None  # callable(np.ndarray) | None

    # ------------------------------------------------------------------ startup
    def load(self) -> None:
        t0 = time.perf_counter()
        s = self.settings
        if not s.job_descriptions_csv.exists():
            raise FileNotFoundError(
                f"{s.job_descriptions_csv} not found. Run the Sprint 1 notebooks "
                "(preprocess.ipynb, embeddings.ipynb) first, or set SCREENING_ARTIFACTS_DIR."
            )

        jobs = pd.read_csv(s.job_descriptions_csv)
        self.categories = jobs["Category"].astype(str).tolist()
        self._category_lookup = {c.upper(): i for i, c in enumerate(self.categories)}
        job_texts = jobs["job_description"].astype(str).tolist()
        self.job_embeddings, self.job_source = self._load_or_encode(
            s.embeddings_dir / "job_embeddings.npy",
            s.embeddings_dir / "job_index.csv",
            expected_ids=self.categories,
            id_column="Category",
            texts=job_texts,
        )

        resumes = None
        if s.resumes_csv.exists():
            resumes = pd.read_csv(s.resumes_csv, usecols=["ID", "Category", "clean_text"])
            self.cohort_ids = resumes["ID"].astype(str).tolist()
            self.cohort_categories = resumes["Category"].astype(str).tolist()
            self.cohort_embeddings, self.cohort_source = self._load_or_encode(
                s.embeddings_dir / "resume_embeddings.npy",
                s.embeddings_dir / "resume_index.csv",
                expected_ids=self.cohort_ids,
                id_column="ID",
                texts=resumes["clean_text"].astype(str).tolist(),
            )
        else:
            logger.warning("%s not found — /v1/cohort/rank will be unavailable.", s.resumes_csv)
            self.cohort_embeddings = np.zeros((0, self.encoder.dim), dtype=np.float32)
            self.cohort_source = "unavailable"

        self.reference = self._load_reference(resumes)

        # Warm-up: the first forward pass is always slow (lazy init, memory
        # allocation). Pay that cost now rather than on a user's first request.
        self.encoder.encode(["warm up the model"], batch_size=1)

        self.ready = True
        logger.info(
            "Service ready in %.1fs: model=%s, %d categories (%s), %d cohort resumes (%s)",
            time.perf_counter() - t0, self.encoder.name, len(self.categories),
            self.job_source, len(self.cohort_ids), self.cohort_source,
        )

    def _load_reference(self, resumes: pd.DataFrame | None) -> ReferenceProfile | None:
        """Drift reference: prefer the profile exported with the artifacts, else build it."""
        s = self.settings
        path = s.reference_profile_json
        if path.exists():
            try:
                prof = ReferenceProfile.from_json(path.read_text())
                if prof.model == self.encoder.name and self.cohort_source.startswith("loaded"):
                    return prof
                logger.info("%s is for model '%s' — rebuilding for '%s'.", path.name, prof.model, self.encoder.name)
            except (ValueError, KeyError, TypeError):
                logger.warning("%s is unreadable — rebuilding the drift reference.", path.name)
        if resumes is None or len(self.cohort_ids) == 0:
            logger.warning("No cohort available — drift monitoring disabled.")
            return None
        return ReferenceProfile.build(
            model=self.encoder.name, categories=self.categories,
            job_embeddings=self.job_embeddings, cohort_embeddings=self.cohort_embeddings,
            text_lengths=resumes["clean_text"].astype(str).str.len().to_numpy(),
            similarity_floor=s.similarity_floor,
        )

    def _observe(self, matrix: np.ndarray) -> None:
        if self.embedding_observer is not None and len(matrix):
            try:
                self.embedding_observer(matrix)
            except Exception:  # monitoring must never break a prediction
                logger.exception("embedding observer failed")

    def _load_or_encode(self, npy_path, index_path, *, expected_ids, id_column, texts):
        """Use saved embeddings if they line up with the CSV *and* match the live model."""
        if npy_path.exists() and index_path.exists():
            saved = np.load(npy_path).astype(np.float32)
            index_ids = pd.read_csv(index_path)[id_column].astype(str).tolist()
            aligned = index_ids == list(expected_ids) and saved.shape == (len(texts), self.encoder.dim)
            if aligned:
                fresh = self.encoder.encode([texts[0]], batch_size=1)[0]
                if float(saved[0] @ fresh) >= SPOT_CHECK_TOLERANCE:
                    return saved, f"loaded {npy_path.name}"
                logger.warning("%s was produced by a different model — re-encoding.", npy_path.name)
            else:
                logger.warning("%s does not line up with the CSV — re-encoding.", npy_path.name)
        logger.info("Encoding %d texts for %s ...", len(texts), npy_path.name)
        vecs = self.encoder.encode(texts, batch_size=self.settings.encode_batch_size)
        return vecs, "encoded at startup"

    # ------------------------------------------------------------- validation
    def _prepare(self, item_id: str, raw: str) -> _Prepared:
        s = self.settings
        if raw is None or not raw.strip():
            return _Prepared(item_id, error_code="EMPTY_TEXT", error_message="Text is empty.")
        if len(raw) > s.max_text_chars:
            return _Prepared(
                item_id, error_code="TEXT_TOO_LONG",
                error_message=f"Text is {len(raw):,} characters; the limit is {s.max_text_chars:,}.",
            )
        cleaned = clean_text(raw)
        if len(cleaned) < s.min_text_chars:
            return _Prepared(
                item_id, error_code="TEXT_TOO_SHORT",
                error_message=f"Text is only {len(cleaned)} characters after cleaning — too short "
                f"for a reliable match (minimum {s.min_text_chars}). Flag for manual review.",
            )
        if letter_count(cleaned) < MIN_LETTERS:
            return _Prepared(
                item_id, error_code="NO_TEXT_CONTENT",
                error_message="Text contains almost no letters — nothing meaningful to score.",
            )
        return _Prepared(item_id, text=cleaned)

    def _check_batch_size(self, n: int) -> None:
        if n > self.settings.max_batch_size:
            raise BatchTooLarge(
                f"Batch has {n} items; the limit is {self.settings.max_batch_size}. "
                "Split it into several requests.",
                details={"limit": self.settings.max_batch_size, "received": n},
            )

    # -------------------------------------------------------------- embedding
    def _embed(self, texts: list[str]) -> tuple[np.ndarray, list[bool]]:
        """Embed many texts with one model call for all cache misses."""
        keys = [text_key(t) for t in texts]
        vectors: list[np.ndarray | None] = [self.cache.get(k) for k in keys]
        cached = [v is not None for v in vectors]

        # De-duplicate misses so identical resumes in one batch are encoded once.
        todo: dict[str, str] = {}
        for k, t, v in zip(keys, texts, vectors):
            if v is None:
                todo.setdefault(k, t)
        if todo:
            try:
                fresh = self.encoder.encode(list(todo.values()), batch_size=self.settings.encode_batch_size)
            except Exception as exc:  # model/runtime failure, not a client error
                logger.exception("encode failed")
                raise InferenceFailed(f"Embedding model failed: {type(exc).__name__}") from exc
            fresh_by_key = dict(zip(todo.keys(), fresh))
            for k, vec in fresh_by_key.items():
                self.cache.put(k, vec)
            vectors = [v if v is not None else fresh_by_key[k] for k, v in zip(keys, vectors)]
        matrix = np.vstack(vectors) if vectors else np.zeros((0, self.encoder.dim), dtype=np.float32)
        return matrix, cached

    # ------------------------------------------------------------- endpoints
    def match(self, items: list[ResumeIn], top_k: int) -> list[MatchResult]:
        """Classify resumes into job categories. Bad items are reported, never raised."""
        self._check_batch_size(len(items))
        prepared = [self._prepare(it.id if it.id is not None else str(i), it.text)
                    for i, it in enumerate(items)]
        good_idx = [i for i, p in enumerate(prepared) if p.text is not None]
        results: list[MatchResult | None] = [None] * len(prepared)

        if good_idx:
            matrix, cached = self._embed([prepared[i].text for i in good_idx])
            self._observe(matrix)
            sims = matrix @ self.job_embeddings.T  # (n_good, n_categories)
            k = min(top_k, len(self.categories))
            for row, i in enumerate(good_idx):
                p = prepared[i]
                order = np.argsort(-sims[row])[:k]
                top = [CategoryScore(category=self.categories[j], similarity=round(float(sims[row, j]), 4))
                       for j in order]
                best = top[0]
                warnings = []
                status = "ok"
                if best.similarity < self.settings.similarity_floor:
                    status = "low_confidence"
                    warnings.append(
                        f"Best match '{best.category}' scored {best.similarity:.3f}, below the "
                        f"{self.settings.similarity_floor} confidence floor — the resume may be outside "
                        "the categories this model covers. Route to manual review."
                    )
                results[i] = MatchResult(
                    id=p.id, status=status, best_category=best.category, similarity=best.similarity,
                    top_matches=top, warnings=warnings, cached=cached[row],
                )

        for i, p in enumerate(prepared):
            if p.text is None:
                results[i] = MatchResult(id=p.id, status="rejected",
                                         warnings=[f"{p.error_code}: {p.error_message}"])
        return results  # same order as the input

    def match_one(self, text: str, top_k: int) -> MatchResult:
        """Single-item variant: an unscoreable input is a 422, not a 'rejected' row."""
        p = self._prepare("0", text)
        if p.text is None:
            raise InputRejected(p.error_code, p.error_message)
        return self.match([ResumeIn(id="0", text=text)], top_k)[0]

    def resolve_job(self, job: JobIn) -> tuple[str, np.ndarray]:
        if job.category:
            idx = self._category_lookup.get(job.category.strip().upper())
            if idx is None:
                suggestions = difflib.get_close_matches(job.category.upper(), self.categories, n=3, cutoff=0.5)
                raise UnknownCategory(
                    f"Unknown category '{job.category}'. See GET /v1/categories for valid values.",
                    details={"suggestions": suggestions},
                )
            return self.categories[idx], self.job_embeddings[idx]
        p = self._prepare("job", job.description)
        if p.text is None:
            raise InputRejected(p.error_code, f"Job description: {p.error_message}")
        matrix, _ = self._embed([p.text])
        return "custom", matrix[0]

    def screen(self, job: JobIn, candidates: list[ResumeIn], top_k: int | None):
        """Rank a set of applicants against one job — the core screening use case."""
        self._check_batch_size(len(candidates))
        label, job_vec = self.resolve_job(job)
        prepared = [self._prepare(c.id if c.id is not None else str(i), c.text)
                    for i, c in enumerate(candidates)]
        good = [p for p in prepared if p.text is not None]
        ranked: list[RankedCandidate] = []
        if good:
            matrix, _ = self._embed([p.text for p in good])
            self._observe(matrix)
            sims = matrix @ job_vec
            order = np.argsort(-sims)
            if top_k is not None:
                order = order[:top_k]
            for rank, j in enumerate(order, start=1):
                sim = round(float(sims[j]), 4)
                low = sim < self.settings.similarity_floor
                ranked.append(RankedCandidate(
                    id=good[j].id, rank=rank, similarity=sim,
                    status="low_confidence" if low else "ok",
                    warnings=[f"Similarity {sim:.3f} is below the {self.settings.similarity_floor} floor."]
                    if low else [],
                ))
        rejected = [
            RankedCandidate(id=p.id, rank=None, status="rejected",
                            warnings=[f"{p.error_code}: {p.error_message}"])
            for p in prepared if p.text is None
        ]
        return label, ranked, rejected

    def rank_cohort(self, job: JobIn, top_k: int) -> tuple[str, list[CohortMatch]]:
        """Rank the indexed Sprint 1 resume cohort against a job."""
        label, job_vec = self.resolve_job(job)
        n = len(self.cohort_ids)
        if n == 0:
            return label, []
        k = min(top_k, self.settings.max_top_k, n)
        sims = self.cohort_embeddings @ job_vec
        top = np.argpartition(-sims, k - 1)[:k]           # O(n) selection...
        top = top[np.argsort(-sims[top])]                  # ...then sort only k items
        return label, [
            CohortMatch(rank=r, resume_id=self.cohort_ids[i], category=self.cohort_categories[i],
                        similarity=round(float(sims[i]), 4))
            for r, i in enumerate(top, start=1)
        ]
