"""Embedding back-ends.

The service only depends on the small ``Encoder`` protocol below, so the real
Sentence-Transformers model can be swapped for a lightweight stand-in in tests
(or a bigger model later) without touching any endpoint code.
"""
from __future__ import annotations

import hashlib
import re
import threading
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    name: str
    dim: int
    max_seq_length: int | None

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised embeddings."""
        ...


class SentenceTransformerEncoder:
    """Production encoder: the same ``all-MiniLM-L6-v2`` used in embeddings.ipynb."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        # Imported lazily so tests and tooling don't pay the torch import cost.
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.max_seq_length = getattr(self._model, "max_seq_length", None)
        # A torch module is not guaranteed to be safe under concurrent forward
        # passes from FastAPI's threadpool; serialise calls. Throughput comes from
        # batching many texts into one call, not from parallel calls.
        self._lock = threading.Lock()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        with self._lock:
            vectors = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,  # cosine similarity == dot product
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return np.asarray(vectors, dtype=np.float32)


class HashingEncoder:
    """Dependency-free bag-of-words hashing encoder.

    Not a substitute for the real model's quality — it exists so the API can be
    exercised in CI or offline (``SCREENING_MODEL_NAME=hashing``) without
    downloading weights. Similar texts still get similar vectors, which is enough
    for tests to assert sensible rankings.
    """

    _TOKEN_RE = re.compile(r"[a-z]{3,}")

    def __init__(self, dim: int = 384) -> None:
        self.name = "hashing"
        self.dim = dim
        self.max_seq_length = None
        self.calls = 0  # how many times encode() ran — lets tests verify caching

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._TOKEN_RE.findall(text.lower()):
                h = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little")
                out[row, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def build_encoder(model_name: str, device: str = "cpu") -> Encoder:
    if model_name == "hashing":
        return HashingEncoder()
    return SentenceTransformerEncoder(model_name, device=device)
