#!/usr/bin/env python3
"""
Fast targeted pytest runner
===========================

Purpose: Run a small set of pytest nodeids with strict per-test timeouts,
capture verbose output, and write a concise summary plus a detailed log file.

Designed to avoid slow collection or tooling overhead and to quickly surface
which specific test is hanging or failing, with enough context to debug.

Usage examples
--------------
python scripts/run_targeted_tests.py \
  --timeout 60 \
  tests/integration/test_pipeline_runner.py::test_runner_unpacks_agent_result \
  tests/integration/test_resilience_features.py::test_cached_fallback_result_is_reused \
  tests/integration/test_pipeline_runner.py::test_timeout_and_redirect_loop_detection

python scripts/run_targeted_tests.py --timeout 45  # uses sensible defaults

Notes
-----
- Uses the current Python interpreter to ensure it runs inside the active venv.
- Kills hanging tests cleanly and marks them as TIMEOUT.
- Writes a detailed combined log to output/targeted_tests.log for later review.
"""

from __future__ import annotations

import argparse
import subprocess as sp
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


DEFAULT_NODEIDS: List[str] = [
    "tests/integration/test_pipeline_runner.py::test_runner_unpacks_agent_result",
    "tests/integration/test_resilience_features.py::test_cached_fallback_result_is_reused",
    "tests/integration/test_pipeline_runner.py::test_timeout_and_redirect_loop_detection",
]


@dataclass
class TestResult:
    nodeid: str
    status: str  # PASS | FAIL | TIMEOUT | ERROR
    duration_s: float
    output: str


def run_one(nodeid: str, timeout: int, show_tb: bool) -> TestResult:
    start = time.perf_counter()
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        nodeid,
        "-vv",
        "--maxfail=1",
        "--disable-warnings",
        "-s",
    ]
    if not show_tb:
        cmd.append("--tb=no")

    try:
        proc = sp.run(cmd, capture_output=True, text=True, timeout=timeout)
        status = "PASS" if proc.returncode == 0 else "FAIL"
        out = proc.stdout + proc.stderr
    except sp.TimeoutExpired as e:
        status = "TIMEOUT"
        def _to_text(buf: bytes | str | None) -> str:
            if buf is None:
                return ""
            if isinstance(buf, (bytes, bytearray)):
                try:
                    return buf.decode()
                except Exception:
                    return buf.decode(errors="replace")
            return buf
        out = _to_text(e.stdout) + _to_text(e.stderr)
    except Exception as exc:  # noqa: BLE001
        status = "ERROR"
        out = str(exc)

    dur = time.perf_counter() - start
    return TestResult(nodeid=nodeid, status=status, duration_s=dur, output=out)


def write_log(results: List[TestResult], log_path: Path) -> None:
    lines: List[str] = []
    lines.append("==== Targeted Test Run Log ====")
    lines.append(f"Total: {len(results)}\n")
    for r in results:
        lines.append(f"--- {r.nodeid}")
        lines.append(f"Status  : {r.status}")
        lines.append(f"Duration: {r.duration_s:.2f}s")
        lines.append("")
        lines.append(r.output.rstrip())
        lines.append("\n" + ("-" * 72) + "\n")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "nodeids",
        nargs="*",
        default=DEFAULT_NODEIDS,
        help="Specific pytest nodeids to run",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-test timeout in seconds (default: 60)",
    )
    ap.add_argument(
        "--tb",
        action="store_true",
        help="Show tracebacks (otherwise use --tb=no for brevity)",
    )
    args = ap.parse_args()

    print(f"Collected {len(args.nodeids)} targeted tests")
    print(f"Using interpreter: {sys.executable}")
    print(f"Per-test timeout: {args.timeout}s\n")

    results: List[TestResult] = []
    status_counts = {"PASS": 0, "FAIL": 0, "TIMEOUT": 0, "ERROR": 0}

    for nid in args.nodeids:
        print(f"▶︎ START {nid}", flush=True)
        res = run_one(nid, args.timeout, args.tb)
        results.append(res)
        status_counts[res.status] += 1
        icon = {
            "PASS": "✅",
            "FAIL": "❌",
            "TIMEOUT": "⏰",
            "ERROR": "💥",
        }[res.status]
        print(f"{icon} {nid} ({res.duration_s:.2f}s) — {res.status}")

    # Summary
    print("\n──── Summary ────")
    for key in ("PASS", "FAIL", "TIMEOUT", "ERROR"):
        if status_counts[key]:
            print(f"{key:8}: {status_counts[key]}")

    # Log details
    log_path = Path("output/targeted_tests.log")
    write_log(results, log_path)
    print(f"\nDetailed log written to: {log_path}")

    exit_code = 0
    if status_counts["TIMEOUT"]:
        exit_code = 3
    elif status_counts["FAIL"]:
        exit_code = 2
    elif status_counts["ERROR"]:
        exit_code = 4
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


