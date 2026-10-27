"""Latency benchmark against a running server.

    python scripts/benchmark.py --url http://localhost:8000 --n 50

Sends real resumes from resumes_clean.csv and reports p50/p95 latency for:
  * single /v1/match calls (cold, i.e. cache misses)
  * the same calls again (warm, i.e. cache hits)
  * one /v1/match/batch call vs. the same items sent one-by-one
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import httpx
import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data_pipeline" / "data" / "processed" / "resumes_clean.csv"


def pct(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def timed(client: httpx.Client, path: str, payload: dict) -> float:
    t0 = time.perf_counter()
    r = client.post(path, json=payload)
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    texts = pd.read_csv(args.csv)["clean_text"].sample(args.n, random_state=7).tolist()
    # Unique suffix per run so "cold" really means cache miss on a long-running server.
    tag = f" run{time.time_ns()}"
    texts = [t + tag for t in texts]

    with httpx.Client(base_url=args.url, timeout=120) as c:
        c.get("/ready").raise_for_status()
        cold = [timed(c, "/v1/match", {"text": t}) for t in texts]
        warm = [timed(c, "/v1/match", {"text": t}) for t in texts]

        batch_texts = [t + " b" for t in texts[: args.batch]]
        one_by_one = sum(timed(c, "/v1/match", {"text": t + " s"}) for t in batch_texts)
        batch_ms = timed(c, "/v1/match/batch",
                         {"items": [{"id": str(i), "text": t} for i, t in enumerate(batch_texts)]})

    print(f"single, cold  (n={len(cold)}): p50={statistics.median(cold):7.1f} ms   p95={pct(cold, 95):7.1f} ms")
    print(f"single, warm  (n={len(warm)}): p50={statistics.median(warm):7.1f} ms   p95={pct(warm, 95):7.1f} ms")
    print(f"{len(batch_texts)} resumes one-by-one: {one_by_one:8.1f} ms total")
    print(f"{len(batch_texts)} resumes in one batch: {batch_ms:8.1f} ms total "
          f"({one_by_one / batch_ms:.1f}x faster)")


if __name__ == "__main__":
    main()
