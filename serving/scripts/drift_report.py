"""Offline drift report from the prediction log files.

    python scripts/drift_report.py --logs logs/ --days 7
    python scripts/drift_report.py --logs logs/ --json drift.json   # machine-readable
    python scripts/drift_report.py --logs logs/ --fail-on alert     # exit 2 → cron / CI alert

Reads ``predictions-YYYY-MM-DD.jsonl`` files written by the API and compares them
with ``reference_profile.json`` — the same checks as ``GET /v1/monitoring/drift``,
but over any period you like rather than the server's in-memory window. (The
embedding-centroid check needs vectors, which are not logged, so it reports
``insufficient_data`` here.)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

SERVING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVING))
from app.monitoring import DriftMonitor, PredictionLog, ReferenceProfile  # noqa: E402
from app.settings import get_settings  # noqa: E402

ICON = {"ok": "OK   ", "warning": "WARN ", "alert": "ALERT", "insufficient_data": "  -  "}


def log_files(folder: Path, days: int | None) -> list[Path]:
    files = sorted(folder.glob("predictions-*.jsonl"))
    if days:
        cutoff = datetime.now(UTC).date() - timedelta(days=days - 1)
        files = [f for f in files if date.fromisoformat(f.stem.split("-", 1)[1]) >= cutoff]
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs", type=Path, default=SERVING / "logs")
    ap.add_argument("--profile", type=Path, default=None,
                    help="reference_profile.json (default: the configured artifacts folder)")
    ap.add_argument("--days", type=int, default=None, help="only the last N days of files")
    ap.add_argument("--endpoint", default=None, help="restrict to match | match_batch | screen")
    ap.add_argument("--min-samples", type=int, default=100)
    ap.add_argument("--json", type=Path, default=None, help="also write the report here")
    ap.add_argument("--fail-on", choices=["warning", "alert"], default=None)
    a = ap.parse_args()

    profile_path = a.profile or get_settings().reference_profile_json
    if not profile_path.exists():
        print(f"No reference profile at {profile_path}. Run scripts/export_artifacts.py first.")
        return 1
    files = log_files(a.logs, a.days)
    if not files:
        print(f"No prediction logs in {a.logs}.")
        return 1
    records = PredictionLog.read_files(files)
    if a.endpoint:
        records = [r for r in records if r.get("endpoint") == a.endpoint]

    report = DriftMonitor(ReferenceProfile.from_json(profile_path.read_text()), a.min_samples).report(records)
    w = report["window"]
    print(f"Drift report — {len(files)} file(s), {w['records']} records ({w['from']} → {w['to']})")
    print(f"Overall: {report['status'].upper()}\n")
    for c in report["checks"]:
        val = "—" if c["value"] is None else f"{c['value']:.3f}"
        print(f"  [{ICON[c['status']]}] {c['name']:<28} {val:>7}   n={c['samples']:<6} {c['detail']}")
    if a.json:
        a.json.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {a.json}")

    order = ["ok", "warning", "alert"]
    if a.fail_on and report["status"] in order and order.index(report["status"]) >= order.index(a.fail_on):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
