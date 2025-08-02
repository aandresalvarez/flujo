#!/usr/bin/env python3
"""
find_problematic_test.py
========================
Run pytest one node at a time (default) or via simple parallel modes
to catch hangs, time-outs, flaky tests, or order-dependent bugs.

───────────────────────────────  CLI  ───────────────────────────────

# 1) Sequential scan (best for dead-locks / resource contention)
python find_problematic_test.py --mode scan

# 2) Process-pool run with 8 workers and 20 s per-test timeout
python find_problematic_test.py --mode pool --workers 8 --timeout 20

# 3) Fastest whole-suite run via pytest-xdist (no per-test stats)
python find_problematic_test.py --mode xdist --workers 0

# 4) Re-run any failure twice (filters out flakes)
python find_problematic_test.py --reruns 2

# 5) Show top-10 slowest tests after the run
python find_problematic_test.py --profile

Optional flags
--------------
--marker  <expr>      pytest -m selector (defaults to fast tests)
--tb                  show tracebacks (otherwise --tb=no for brevity)
--workers <N>         worker count (0 ⇒ CPU count) for pool/xdist
--timeout <sec>       kill a single test after N seconds (default 30s)
--reruns <N>          how many times to retry a failing test
--profile             print slowest-10 tests and their runtimes

Exit codes
----------
0  all tests pass
2  assertion failures
3  time-outs / hangs
4  infra or collection errors
5  fails in parallel but passes serially (order-fail)
6  flaky tests that pass after re-run
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import enum
import os
import subprocess as sp
import sys
import time
from typing import Dict, List, Tuple

# ──────────────────────────── ENUMS & UTILS ───────────────────────── #

class Status(enum.IntEnum):
    PASS = 0
    FAIL = 1
    TIMEOUT = 2
    ERROR = 3
    ORDER_FAIL = 5
    FLAKY = 6


def color(txt: str, code: str, enable: bool) -> str:
    """Return coloured text if stdout is a TTY."""
    return f"\x1b[{code}m{txt}\x1b[0m" if enable else txt


def to_str(buf: bytes | str | None) -> str:
    """Decode bytes (from TimeoutExpired) into str; treat None as ''."""
    if buf is None:
        return ""
    return buf.decode(errors="replace") if isinstance(buf, (bytes, bytearray)) else buf


def announce_start(nodeid: str, tty: bool) -> None:
    """Print which node is about to run—helps pinpoint hard hangs."""
    print(color(f"▶︎ START {nodeid}", "34", tty), flush=True)


# ───────────────────────── SUBPROCESS WRAPPER ────────────────────── #

def run_pytest(cmd: List[str], timeout: int | None) -> Tuple[Status, str, float]:
    """
    Execute *cmd*; classify returncode & handle time-outs.
    Returns (Status, combined_stdout_stderr, duration_seconds).
    """
    start = time.perf_counter()
    try:
        proc = sp.run(cmd, capture_output=True, text=True, timeout=timeout)
        stat = Status.PASS if proc.returncode == 0 else Status.FAIL
        out  = proc.stdout + proc.stderr
    except sp.TimeoutExpired as e:
        stat = Status.TIMEOUT
        out  = to_str(e.stdout) + to_str(e.stderr)
    except Exception as exc:                                 # noqa: BLE001
        stat = Status.ERROR
        out  = str(exc)
    dur = time.perf_counter() - start
    return stat, out, dur


# ───────────────────────── SINGLE-NODE EXECUTOR ──────────────────── #

def exec_one(nodeid: str,
             timeout: int,
             reruns: int,
             show_tb: bool,
             tty: bool) -> Tuple[str, Status, str, float]:
    """
    Run *nodeid* once (optionally re-running on failure).
    Returns (nodeid, final_status, detail, duration).
    """
    announce_start(nodeid, tty)
    base = ["uv", "run", "pytest", nodeid, "-v"]
    if not show_tb:
        base.append("--tb=no")

    def attempt() -> Tuple[Status, str, float]:
        return run_pytest(base, timeout)

    status, detail, dur = attempt()

    # flake filter
    if status != Status.PASS and reruns:
        for _ in range(reruns):
            r_status, r_detail, r_dur = attempt()
            detail += r_detail
            dur += r_dur
            if r_status == Status.PASS:
                status = Status.FLAKY
                break

    if status == Status.TIMEOUT:
        detail = f"[TIMEOUT] {nodeid}\n" + detail
    return nodeid, status, detail, dur


# ───────────────────────────── MAIN ──────────────────────────────── #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("scan", "pool", "xdist"), default="scan")
    ap.add_argument("--workers", type=int, default=0,
                    help="worker count for pool/xdist (0 ⇒ CPU count)")
    ap.add_argument("--timeout", type=int, default=30,
                    help="per-test timeout in seconds")
    ap.add_argument("--marker", default="not slow and not serial and not benchmark",
                    help="pytest -m expression for test selection")
    ap.add_argument("--reruns", type=int, default=0,
                    help="re-run failing test up to N times")
    ap.add_argument("--tb", action="store_true",
                    help="show tracebacks (default off)")
    ap.add_argument("--profile", action="store_true",
                    help="print top-10 slowest tests")
    args = ap.parse_args()
    tty = sys.stdout.isatty()

    # ---- collect nodeids -------------------------------------------------- #
    coll_cmd = ["uv", "run", "pytest", "tests/", "-q", "--collect-only", "-m", args.marker]
    coll_status, coll_out, _ = run_pytest(coll_cmd, 60)
    if coll_status != Status.PASS:
        print(color("Collection failed:\n" + coll_out, "31", tty))
        sys.exit(4)
    nodeids = [ln.strip() for ln in coll_out.splitlines()
               if ln and not ln.startswith("=")]
    print(color(f"Collected {len(nodeids)} tests  (mode={args.mode})", "36", tty))

    stats = {s: 0 for s in Status}
    durations: Dict[str, float] = {}
    failures: List[Tuple[str, Status, str]] = []

    # ---- choose execution mode ------------------------------------------- #
    if args.mode == "scan":
        iterator = (exec_one(n, args.timeout, args.reruns, args.tb, tty)
                    for n in nodeids)

    elif args.mode == "pool":
        max_w = os.cpu_count() if args.workers <= 0 else args.workers
        with cf.ProcessPoolExecutor(max_workers=max_w) as ex:
            iterator = ex.map(
                lambda n: exec_one(n, args.timeout, args.reruns, args.tb, False),
                nodeids
            )

    else:  # xdist (fastest, no per-test stats)
        workers = "auto" if args.workers <= 0 else str(args.workers)
        xcmd = ["uv", "run", "pytest", "tests/", "-m", args.marker,
                "-n", workers, "--dist=loadscope"]
        x_status, x_out, _ = run_pytest(xcmd, None)
        print(x_out)
        sys.exit(x_status.value)

    # ---- run loop --------------------------------------------------------- #
    icons = {Status.PASS: "✅", Status.FAIL: "❌", Status.TIMEOUT: "⏰",
             Status.ERROR: "💥", Status.FLAKY: "🤷", Status.ORDER_FAIL: "🔀"}

    for nid, st, detail, dur in iterator:
        print(f"{icons.get(st, '•')} {nid} ({dur:.2f}s)")
        stats[st] += 1
        durations[nid] = dur
        if st != Status.PASS:
            failures.append((nid, st, detail))

    # ---- summary ---------------------------------------------------------- #
    print("\n──── Summary ────")
    for s in Status:
        if stats[s]:
            print(color(f"{s.name:10}: {stats[s]}", "33" if s else "32", tty))

    if args.profile:
        top = sorted(durations.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print(color("\nTop-10 slowest tests:", "36", tty))
        for nid, dur in top:
            print(f"{dur:6.2f}s  {nid}")

    if failures:
        print(color("\nDetailed reports:", "31", tty))
        for nid, st, det in failures:
            print(color(f"[{st.name}] {nid}", "31", tty))
            print(det.rstrip() + "\n" + "-"*60)

    # ---- exit codes ------------------------------------------------------- #
    if stats[Status.TIMEOUT]:
        sys.exit(3)
    if stats[Status.FAIL]:
        sys.exit(2)
    if stats[Status.ERROR]:
        sys.exit(4)
    if stats[Status.FLAKY]:
        sys.exit(6)
    sys.exit(0)


if __name__ == "__main__":
    main()
