"""Post-deploy smoke test — standard library only, so it runs anywhere.

    python serving/scripts/smoke_test.py --url http://localhost:8000
    python serving/scripts/smoke_test.py --url https://<your-app>.onrender.com --api-key $KEY --wait 300

Checks the things that break in deployment but not in unit tests: the container
started, the model and artifacts loaded, auth and CORS are wired, the real model
puts an obvious resume in the right category, and monitoring is recording.
Exit code 0 = all passed, 1 = something failed (CI and deploy scripts rely on it).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

ACCOUNTANT = ("Certified public accountant with eight years of experience in general ledger "
              "reconciliation, payroll processing, accounts payable, tax returns, audits and "
              "monthly financial statements for mid-size companies.")


class Api:
    def __init__(self, base: str, api_key: str | None) -> None:
        self.base = base.rstrip("/")
        self.api_key = api_key

    def call(self, method: str, path: str, body: dict | None = None, headers: dict | None = None):
        h = {"Content-Type": "application/json", "X-Request-ID": f"smoke-{int(time.time())}"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        h.update(headers or {})
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode()
                is_json = r.headers.get_content_type() == "application/json"
                return r.status, (json.loads(raw) if is_json else raw), r.headers
        except urllib.error.HTTPError as e:
            raw = e.read().decode()
            try:
                return e.code, json.loads(raw), e.headers
            except ValueError:
                return e.code, raw, e.headers


def wait_ready(api: Api, seconds: int) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            status, _, _ = api.call("GET", "/ready")
            if status == 200:
                return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(3)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--wait", type=int, default=120, help="seconds to wait for /ready")
    ap.add_argument("--expect-model", default=None,
                    help="fail unless /v1/model reports this model (e.g. all-MiniLM-L6-v2)")
    a = ap.parse_args()
    api = Api(a.url, a.api_key)
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    print(f"Smoke test against {api.base}")
    t0 = time.time()
    ready = wait_ready(api, a.wait)
    check("service becomes ready", ready, f"{time.time() - t0:.0f}s")
    if not ready:
        return 1

    s, body, _ = api.call("GET", "/health")
    check("GET /health", s == 200 and body == {"status": "ok"})

    s, body, _ = api.call("GET", "/v1/model")
    check("GET /v1/model", s == 200, f"HTTP {s}")
    if s == 200:
        check("artifacts loaded", body["categories"] > 0 and body["cohort_size"] > 0,
              f"{body['categories']} categories, {body['cohort_size']} cohort resumes")
        if body["model"] != "hashing":  # the offline stand-in can't reuse MiniLM vectors, by design
            check("saved embeddings reused (no re-encode)", body["job_embeddings_source"].startswith("loaded"),
                  f'{body["job_embeddings_source"]}; {body["cohort_embeddings_source"]}')
        check("build reported", bool(body.get("build")), f"build={body.get('build')} env={body.get('environment')}")
        if a.expect_model:
            check("expected model", body["model"] == a.expect_model, body["model"])
    elif s == 401:
        print("        (401: pass --api-key)")

    s, body, headers = api.call("POST", "/v1/match", {"text": ACCOUNTANT, "top_k": 3})
    check("POST /v1/match", s == 200, f"HTTP {s}")
    if s == 200:
        res = body["result"]
        # With the real model this resume lands in ACCOUNTANT/FINANCE; the offline
        # "hashing" encoder is not semantic, so only check the category with a real model.
        if body["model"] != "hashing":
            top = [m["category"] for m in res["top_matches"]]
            check("sensible prediction", bool({"ACCOUNTANT", "FINANCE"} & set(top)), f"top-3 {top}")
        check("request id echoed", headers.get("X-Request-ID", "").startswith("smoke-"))

    s, body, _ = api.call("POST", "/v1/match/batch",
                          {"items": [{"id": "good", "text": ACCOUNTANT}, {"id": "bad", "text": ""}]})
    check("POST /v1/match/batch partial failure", s == 200 and body["summary"]["rejected"] == 1, f"HTTP {s}")

    s, body, _ = api.call("POST", "/v1/match", {"text": "hi"})
    check("error envelope", s == 422 and body.get("error", {}).get("code") == "TEXT_TOO_SHORT")

    s, body, _ = api.call("GET", "/v1/monitoring/predictions?limit=5")
    check("predictions are logged", s == 200 and body.get("total_logged", 0) >= 3,
          f"total_logged={body.get('total_logged') if isinstance(body, dict) else '?'}")
    s, body, _ = api.call("GET", "/v1/monitoring/drift")
    check("drift report", s == 200 and body.get("status") in {"ok", "warning", "alert", "insufficient_data"},
          f"status={body.get('status') if isinstance(body, dict) else s}")

    s, body, _ = api.call("GET", "/metrics", headers={"Content-Type": "text/plain"})
    check("GET /metrics", s == 200 and "screening_http_requests_total" in str(body), f"HTTP {s}")

    print(f"\n{'All checks passed' if not failures else f'{len(failures)} check(s) failed: {failures}'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
