"""Applicant Screening API — Python client for the Backend squad.

    from screening_client import ScreeningClient, ScreeningApiError

    with ScreeningClient("http://localhost:8000") as api:
        ranked = api.screen({"category": "ACCOUNTANT"}, [{"id": "42", "text": resume}])

Features: connection pooling (one ``httpx.Client``), timeouts, automatic retry
while the model is loading (503) or on a dropped connection, transparent
chunking of large batches, and a typed exception carrying the server's error
``code`` and ``request_id``. Pass ``http=`` to reuse an existing client — e.g. a
FastAPI ``TestClient`` in tests or notebooks.

Requires: ``pip install httpx``
"""
from __future__ import annotations

import time
from typing import Any

import httpx


class ScreeningApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: Any = None, request_id: str | None = None):
        super().__init__(f"[{status} {code}] {message}")
        self.status, self.code, self.message = status, code, message
        self.details, self.request_id = details, request_id


class ScreeningClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_batch_size: int = 64,
        retries: int = 2,
        http: httpx.Client | None = None,
    ) -> None:
        headers = {"X-API-Key": api_key} if api_key else {}
        self._http = http or httpx.Client(base_url=base_url, timeout=timeout, headers=headers)
        if http is not None and api_key:
            self._http.headers.update(headers)
        self._owns_http = http is None
        self.max_batch_size = max_batch_size
        self.retries = retries

    # -- plumbing -----------------------------------------------------------
    def _request(self, method: str, path: str, json: dict | None = None) -> dict:
        for attempt in range(self.retries + 1):
            try:
                resp = self._http.request(method, path, json=json)
            except httpx.TransportError:
                if attempt == self.retries:
                    raise
                time.sleep(0.5 * (attempt + 1))
                continue
            if resp.is_success:
                return resp.json()
            if resp.status_code == 503 and attempt < self.retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            try:
                err = resp.json()["error"]
            except Exception:
                err = {"code": "HTTP_ERROR", "message": resp.text}
            raise ScreeningApiError(resp.status_code, err.get("code"), err.get("message"),
                                    err.get("details"), err.get("request_id"))
        raise RuntimeError("unreachable")

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> ScreeningClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- API ----------------------------------------------------------------
    def health(self) -> dict:
        return self._request("GET", "/health")

    def categories(self) -> list[str]:
        return self._request("GET", "/v1/categories")["categories"]

    def model_info(self) -> dict:
        return self._request("GET", "/v1/model")

    def match(self, text: str, top_k: int = 3) -> dict:
        return self._request("POST", "/v1/match", {"text": text, "top_k": top_k})["result"]

    def match_batch(self, items: list[dict], top_k: int = 3) -> list[dict]:
        """items: [{"id": ..., "text": ...}]. Chunked to the server's batch limit; order preserved."""
        out: list[dict] = []
        for start in range(0, len(items), self.max_batch_size):
            chunk = [{"id": str(it.get("id", start + i)), "text": it["text"]}
                     for i, it in enumerate(items[start:start + self.max_batch_size])]
            out.extend(self._request("POST", "/v1/match/batch", {"items": chunk, "top_k": top_k})["results"])
        return out

    def screen(self, job: dict, candidates: list[dict], top_k: int | None = None) -> dict:
        payload: dict = {"job": job, "candidates": candidates}
        if top_k is not None:
            payload["top_k"] = top_k
        return self._request("POST", "/v1/screen", payload)

    def rank_cohort(self, job: dict, top_k: int = 10) -> dict:
        return self._request("POST", "/v1/cohort/rank", {"job": job, "top_k": top_k})


if __name__ == "__main__":  # quick smoke test against a running server
    with ScreeningClient() as api:
        print(api.health())
        print(api.categories()[:5], "...")
        print(api.match("Certified accountant experienced in payroll, tax returns and general ledger close."))
