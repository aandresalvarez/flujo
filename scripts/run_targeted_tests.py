#!/usr/bin/env python3
"""
Enhanced pytest runner with exploration, timeouts, parallelism, and rich logs.

Key features
------------
- Strict per-test timeouts via pytest-timeout (keeps test-level stacks) +
  an outer subprocess failsafe timeout.
- Optional faulthandler stack dump before kill (UNIX).
- Parallel targeted runs (--workers) while keeping one-pytest-per-test isolation.
- Solid discovery (no 'tests/' prefixing guess), supports -m and -k filters.
- Better error extraction from pytest output.
- JSON + text logs, optional "only failures" in detailed text log.
- Fail-fast mode stops after first failure across the whole run.

Install (dev):
  uv add --dev pytest-timeout psutil
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import platform
import re
import shlex
import signal
import subprocess as sp
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # optional


DEFAULT_NODEIDS: List[str] = [
    "tests/integration/test_pipeline_runner.py::test_runner_unpacks_agent_result",
    "tests/integration/test_resilience_features.py::test_cached_fallback_result_is_reused",
    "tests/integration/test_pipeline_runner.py::test_timeout_and_redirect_loop_detection",
]


@dataclass
class TestResult:
    nodeid: str
    status: str  # PASS | FAIL | TIMEOUT | ERROR | PASS_LINGER
    duration_s: float
    output: str
    error_details: Optional[str] = None


# -----------------------------
# Helpers
# -----------------------------


def _supports_sigusr2() -> bool:
    return hasattr(signal, "SIGUSR2") and platform.system() != "Windows"


def _kill_process_tree(proc: sp.Popen) -> None:
    """Best-effort kill of a process and its children."""
    if psutil is not None:
        try:
            p = psutil.Process(proc.pid)
            children = p.children(recursive=True)
            for c in children:
                try:
                    c.terminate()
                except Exception:
                    pass
            _, alive = psutil.wait_procs(children, timeout=1.0)
            for c in alive:
                try:
                    c.kill()
                except Exception:
                    pass
            try:
                p.terminate()
            except Exception:
                pass
            try:
                p.wait(timeout=1.0)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        except Exception:
            # Fallback below
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
    else:
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _classify_linger(stdout_stderr: str) -> bool:
    """Detect tests that pass but pytest process didn't exit quickly."""
    text = stdout_stderr.lower()
    # Look for a clean 'passed' summary and no failure keywords.
    if re.search(r"\b\d+\s+passed\b", text) and not re.search(
        r"\b(failed|error|timeout|interrupt(ed)?)\b", text
    ):
        return True
    return False


_SHORT_SUMMARY_RE = re.compile(r"=+ short test summary info =+\n(?P<body>.+?)(?:\n=+|$)", re.S)
_LAST_E_BLOCK_RE = re.compile(r"(?P<block>(?:^E\s+.*\n?)+)", re.M)
_ASSERT_LINE_RE = re.compile(r"AssertionError.*", re.I)


def _extract_error_details(output: str) -> Optional[str]:
    """Pull the most relevant part of pytest's failure output."""
    # Prefer the last contiguous "E   ..." block
    e_blocks = list(_LAST_E_BLOCK_RE.finditer(output))
    if e_blocks:
        return e_blocks[-1].group("block").strip()

    # Fall back to short test summary info
    m = _SHORT_SUMMARY_RE.search(output)
    if m:
        return m.group("body").strip()

    # Grab a direct AssertionError line if present
    a = _ASSERT_LINE_RE.search(output)
    if a:
        return a.group(0).strip()

    # Nothing obvious
    return None


def _safe_decode(buf: bytes | str | None) -> str:
    if buf is None:
        return ""
    if isinstance(buf, (bytes, bytearray)):
        try:
            return buf.decode()
        except Exception:
            return buf.decode(errors="replace")
    return buf


def _env_with_options(disable_autoload: bool) -> dict:
    env = dict(os.environ)
    if disable_autoload:
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    # Enable faulthandler printing by default; pytest also enables its plugin.
    env.setdefault("PYTHONFAULTHANDLER", "1")
    return env


# -----------------------------
# Core execution
# -----------------------------


def run_one(
    nodeid: str,
    per_test_timeout: int,
    outer_timeout: int,
    show_tb: bool,
    pytest_args: Sequence[str],
    disable_autoload: bool,
    faulthandler_timeout: Optional[int],
    markers: Optional[str],
    kexpr: Optional[str],
) -> TestResult:
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
        # per-test timeout via plugin (preserves stacks)
        f"--timeout={per_test_timeout}",
    ]
    # Print stacks if whole run stalls (if supported/desired)
    if faulthandler_timeout is not None:
        cmd.append(f"--faulthandler-timeout={faulthandler_timeout}")

    # Only force-load timeout/asyncio plugins when plugin autoload is disabled.
    # Otherwise, autoload will bring them in and double-registration would error.
    if disable_autoload:
        cmd += ["-p", "pytest_timeout", "-p", "pytest_asyncio.plugin"]

    if not show_tb:
        cmd.append("--tb=no")

    if markers:
        cmd += ["-m", markers]
    if kexpr:
        cmd += ["-k", kexpr]
    if pytest_args:
        cmd += list(pytest_args)

    # We run one nodeid per subprocess for isolation.
    # Use Popen to allow a pre-kill signal for faulthandler dumps.
    env = _env_with_options(disable_autoload)
    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, text=True, env=env)

    try:
        stdout, stderr = proc.communicate(timeout=outer_timeout)
        rc = proc.returncode
        out = (stdout or "") + (stderr or "")
        status = "PASS" if rc == 0 else "FAIL"
        err = _extract_error_details(out) if status == "FAIL" else None
    except sp.TimeoutExpired:
        # Try to trigger faulthandler dump before killing (UNIX).
        if _supports_sigusr2():
            try:
                os.kill(proc.pid, signal.SIGUSR2)  # request stack dump
                time.sleep(1.0)  # give it a moment to print
            except Exception:
                pass

        # Now kill the process tree
        _kill_process_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=1.0)
        except Exception:
            stdout, stderr = "", ""

        out = _safe_decode(stdout) + _safe_decode(stderr)
        status = "PASS_LINGER" if _classify_linger(out) else "TIMEOUT"
        err = f"TEST TIMED OUT — outer timeout {outer_timeout}s (per-test timeout {per_test_timeout}s)"
    except Exception as exc:  # noqa: BLE001
        _kill_process_tree(proc)
        out = f"Runner error: {exc}"
        status = "ERROR"
        err = f"Execution error: {exc}"

    dur = time.perf_counter() - start
    return TestResult(nodeid=nodeid, status=status, duration_s=dur, output=out, error_details=err)


def discover_tests(
    markers: Optional[str],
    kexpr: Optional[str],
    pytest_args: Sequence[str],
    disable_autoload: bool,
    collect_timeout: int = 180,
) -> List[str]:
    """Return all collected nodeids honoring -m and -k filters."""
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    if disable_autoload:
        # Ensure required plugins are available during collection
        cmd += ["-p", "pytest_asyncio.plugin"]
    if markers:
        cmd += ["-m", markers]
    if kexpr:
        cmd += ["-k", kexpr]
    if pytest_args:
        cmd += list(pytest_args)

    env = _env_with_options(disable_autoload)
    try:
        res = sp.run(cmd, capture_output=True, text=True, timeout=collect_timeout, env=env)
    except sp.TimeoutExpired:
        print("❌ Test discovery timed out.")
        return []

    if res.returncode != 0:
        # Some plugins print to stderr on collect; show a hint
        print("⚠️  Test discovery returned non-zero. Output may include plugin chatter.")
        # Still try to parse stdout

    nodeids: List[str] = []
    for raw in (res.stdout or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("="):
            continue
        if "::" in line:
            nodeids.append(line)
            continue
        # Accept file summary lines like "path/to/test_file.py: N" by extracting the path
        m = re.match(r"^(?P<path>.+\.py)(?::\s*\d+)?$", line)
        if m:
            nodeids.append(m.group("path"))

    # Prefer official test directories only (avoid accidental debug files elsewhere)
    filtered: List[str] = []
    for nid in nodeids:
        # Keep only items under a tests/ directory
        if "/tests/" in nid or nid.startswith("tests/") or nid.startswith("flujo/tests/"):
            filtered.append(nid)
    # Do NOT fallback to unfiltered paths; if nothing matches, return empty to avoid stray files
    return filtered


# -----------------------------
# Logging
# -----------------------------


def write_text_log(results: List[TestResult], path: Path, show_only_failures: bool) -> None:
    lines: List[str] = []
    lines.append("==== Controlled Test Run Log ====")
    lines.append(f"Total Tests: {len(results)}")

    # Counts
    keys = ["PASS", "PASS_LINGER", "FAIL", "TIMEOUT", "ERROR"]
    counts = {k: 0 for k in keys}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    lines.append(" " + " | ".join(f"{k}: {counts.get(k,0)}" for k in keys))
    lines.append("")

    to_show = [r for r in results if r.status != "PASS"] if show_only_failures else results
    for r in to_show:
        lines.append(f"--- {r.nodeid}")
        lines.append(f"Status  : {r.status}")
        lines.append(f"Duration: {r.duration_s:.2f}s")
        if r.error_details:
            lines.append("")
            lines.append("ERROR DETAILS:")
            lines.append(r.error_details)
        lines.append("")
        lines.append("FULL OUTPUT:")
        lines.append(r.output.rstrip())
        lines.append("\n" + ("-" * 72) + "\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def write_failure_summary(results: List[TestResult], path: Path) -> None:
    failing = [r for r in results if r.status in ("FAIL", "TIMEOUT", "ERROR")]
    if not failing:
        path.write_text("🎉 All tests passed successfully!\n")
        return

    lines = ["🚨 FAILURE SUMMARY", ""]
    for r in failing:
        lines.append(f"❌ {r.nodeid}")
        lines.append(f"   Status  : {r.status}")
        lines.append(f"   Duration: {r.duration_s:.2f}s")
        if r.error_details:
            lines.append(f"   Error   : {r.error_details}")
        lines.append("")
    lines.append(f"Total failures: {len(failing)}")
    path.write_text("\n".join(lines))


def write_json(results: List[TestResult], path: Path) -> None:
    data = [asdict(r) for r in results]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# -----------------------------
# Runner
# -----------------------------


def _auto_workers() -> int:
    try:
        n = os.cpu_count() or 4
    except Exception:
        n = 4
    # Keep it conservative to avoid overloading local dev
    return max(1, min(8, (n + 1) // 2))


def _run_serial(
    nodeids: List[str],
    per_test_timeout: int,
    outer_timeout: int,
    show_tb: bool,
    pytest_args: Sequence[str],
    disable_autoload: bool,
    fail_fast: bool,
    faulthandler_timeout: Optional[int],
    markers: Optional[str],
    kexpr: Optional[str],
) -> Tuple[List[TestResult], Optional[TestResult]]:
    results: List[TestResult] = []
    first_failure: Optional[TestResult] = None
    for i, nid in enumerate(nodeids, 1):
        print(f"▶︎ [{i}/{len(nodeids)}] START {nid}", flush=True)
        res = run_one(
            nid,
            per_test_timeout,
            outer_timeout,
            show_tb,
            pytest_args,
            disable_autoload,
            faulthandler_timeout,
            markers,
            kexpr,
        )
        results.append(res)
        icon = {"PASS": "✅", "PASS_LINGER": "✅⏳", "FAIL": "❌", "TIMEOUT": "⏰", "ERROR": "💥"}[
            res.status
        ]
        print(f"{icon} {nid} ({res.duration_s:.2f}s) — {res.status}")
        if res.status in ("FAIL", "TIMEOUT", "ERROR"):
            if res.error_details:
                print(f"   💡 {res.error_details}")
            if first_failure is None:
                first_failure = res
            if fail_fast:
                print(f"\n🚨 FAIL-FAST: Stopping after first failure at test {i}/{len(nodeids)}")
                print(f"⏹️  Remaining tests ({len(nodeids) - i}) will not be executed")
                break
    return results, first_failure


def _run_parallel(
    nodeids: List[str],
    workers: int,
    per_test_timeout: int,
    outer_timeout: int,
    show_tb: bool,
    pytest_args: Sequence[str],
    disable_autoload: bool,
    fail_fast: bool,
    faulthandler_timeout: Optional[int],
    markers: Optional[str],
    kexpr: Optional[str],
) -> Tuple[List[TestResult], Optional[TestResult]]:
    print(f"🧵 Parallel mode: {workers} workers")
    results: List[TestResult] = []
    first_failure: Optional[TestResult] = None

    # ThreadPool is fine here; each task is a subprocess call.
    pending = {}
    started = 0
    done = 0

    def submit_one(exe: cf.Executor, nid: str):
        nonlocal started
        started += 1
        print(f"▶︎ [{started}/{len(nodeids)}] START {nid}", flush=True)
        fut = exe.submit(
            run_one,
            nid,
            per_test_timeout,
            outer_timeout,
            show_tb,
            tuple(pytest_args),
            disable_autoload,
            faulthandler_timeout,
            markers,
            kexpr,
        )
        pending[fut] = nid

    with cf.ThreadPoolExecutor(max_workers=workers) as exe:
        # Prime the pool
        for nid in nodeids[:workers]:
            submit_one(exe, nid)

        next_index = workers
        while pending:
            done_set, _ = cf.wait(pending.keys(), return_when=cf.FIRST_COMPLETED)
            for fut in done_set:
                nid = pending.pop(fut)
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001
                    res = TestResult(
                        nodeid=nid,
                        status="ERROR",
                        duration_s=0.0,
                        output=str(exc),
                        error_details=str(exc),
                    )
                results.append(res)
                done += 1
                icon = {
                    "PASS": "✅",
                    "PASS_LINGER": "✅⏳",
                    "FAIL": "❌",
                    "TIMEOUT": "⏰",
                    "ERROR": "💥",
                }[res.status]
                print(f"{icon} {nid} ({res.duration_s:.2f}s) — {res.status}")
                if res.status in ("FAIL", "TIMEOUT", "ERROR") and res.error_details:
                    print(f"   💡 {res.error_details}")

                if res.status in ("FAIL", "TIMEOUT", "ERROR") and first_failure is None:
                    first_failure = res
                    if fail_fast:
                        print("\n🚨 FAIL-FAST: Stopping after first failure (parallel mode)")
                        # Drain remaining futures but don't start new ones
                        for f in list(pending.keys()):
                            f.cancel()
                        pending.clear()
                        break

                # Start next if any remain
                if next_index < len(nodeids) and not fail_fast:
                    submit_one(exe, nodeids[next_index])
                    next_index += 1

            # If we break due to fail_fast, exit loop
            if fail_fast and first_failure is not None and not pending:
                break

    return results, first_failure


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "nodeids",
        nargs="*",
        default=[],
        help="Specific pytest nodeids to run (if not using --full-suite)",
    )
    ap.add_argument(
        "--full-suite",
        action="store_true",
        help="Run the full test suite instead of specific tests",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-test timeout (pytest-timeout) in seconds (default: 60)",
    )
    ap.add_argument(
        "--outer-timeout",
        type=int,
        default=None,
        help="Outer subprocess timeout in seconds (default: timeout * 3)",
    )
    ap.add_argument(
        "--tb", action="store_true", help="Show tracebacks (otherwise use --tb=no for brevity)"
    )
    ap.add_argument(
        "--faulthandler-timeout",
        type=int,
        default=None,
        help="Enable pytest faulthandler timeout (seconds) if supported",
    )
    ap.add_argument(
        "--markers", type=str, help="Pytest -m expression (e.g., 'not slow and not serial')"
    )
    ap.add_argument("--kexpr", type=str, help="Pytest -k filter expression (substring/expr match)")
    ap.add_argument("--pytest-args", type=str, help='Extra pytest args (e.g., "--maxfail=1 -q -s")')
    ap.add_argument(
        "--show-only-failures", action="store_true", help="In detailed log, show only failing tests"
    )
    ap.add_argument(
        "--fail-fast", action="store_true", help="Stop immediately after the first failure"
    )
    ap.add_argument(
        "--workers",
        type=str,
        default="1",
        help="Number of concurrent test processes (int or 'auto'). Default: 1",
    )
    ap.add_argument(
        "--disable-plugin-autoload",
        action="store_true",
        help="Set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 for faster/cleaner runs",
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="Limit number of discovered tests (for sampling)"
    )
    # Optional two-phase execution for full suite: run non-slow tests first, then slow tests
    ap.add_argument(
        "--split-slow",
        dest="split_slow",
        action="store_true",
        default=True,
        help="Run slow-marked tests in a separate phase (default: enabled)",
    )
    ap.add_argument(
        "--no-split-slow",
        dest="split_slow",
        action="store_false",
        help="Disable separate slow phase; run all tests together",
    )
    ap.add_argument(
        "--slow-workers",
        type=str,
        default="1",
        help="Workers for the slow phase (int or 'auto'). Default: 1",
    )
    ap.add_argument(
        "--slow-timeout",
        type=int,
        default=None,
        help="Per-test timeout (seconds) for slow phase (default: 2x --timeout)",
    )
    args = ap.parse_args()

    # Parse pytest args safely
    pytest_args: List[str] = shlex.split(args.pytest_args) if args.pytest_args else []

    # Per-test + outer timeouts
    per_test_timeout = int(args.timeout)
    outer_timeout = (
        int(args.outer_timeout)
        if args.outer_timeout is not None
        else max(per_test_timeout * 3, per_test_timeout + 60)
    )

    # Determine tests to run
    if args.full_suite and args.split_slow:
        print("🔍 Discovering tests (non-slow and slow phases)...")
        base_markers = (args.markers or "").strip()

        def _and(expr: str) -> str:
            if not expr:
                return base_markers
            if not base_markers:
                return expr
            return f"({base_markers}) and {expr}"

        nodeids_fast = discover_tests(
            _and("not slow"), args.kexpr, pytest_args, args.disable_plugin_autoload
        )
        nodeids_slow = discover_tests(
            _and("(slow or veryslow)"), args.kexpr, pytest_args, args.disable_plugin_autoload
        )
        # Apply sampling limit to fast group only (for quick smoke runs)
        if args.limit:
            nodeids_fast = nodeids_fast[: args.limit]
        print(f"📋 Discovered {len(nodeids_fast)} non-slow and {len(nodeids_slow)} slow tests")
        # Run fast phase
        print("\n=== Phase 1/2: Running non-slow tests ===")
        # Workers for fast phase
        if args.workers.strip().lower() == "auto":
            workers = _auto_workers()
        else:
            try:
                workers = max(1, int(args.workers))
            except Exception:
                workers = 1
        parallel = workers > 1 and len(nodeids_fast) > 1
        if nodeids_fast:
            if parallel:
                results_fast, first_failure_fast = _run_parallel(
                    nodeids_fast,
                    workers,
                    per_test_timeout,
                    outer_timeout,
                    args.tb,
                    pytest_args,
                    args.disable_plugin_autoload,
                    args.fail_fast,
                    args.faulthandler_timeout,
                    _and("not slow"),
                    args.kexpr,
                )
            else:
                results_fast, first_failure_fast = _run_serial(
                    nodeids_fast,
                    per_test_timeout,
                    outer_timeout,
                    args.tb,
                    pytest_args,
                    args.disable_plugin_autoload,
                    args.fail_fast,
                    args.faulthandler_timeout,
                    _and("not slow"),
                    args.kexpr,
                )
        else:
            results_fast, first_failure_fast = ([], None)

        # If fail-fast triggered, stop before slow phase
        if args.fail_fast and first_failure_fast is not None:
            results = results_fast
            first_failure = first_failure_fast
        else:
            # Run slow phase with reduced workers and longer timeout
            print("\n=== Phase 2/2: Running slow tests ===")
            if args.slow_workers.strip().lower() == "auto":
                slow_workers = max(1, _auto_workers() // 2)
            else:
                try:
                    slow_workers = max(1, int(args.slow_workers))
                except Exception:
                    slow_workers = 1
            slow_timeout = (
                int(args.slow_timeout)
                if args.slow_timeout is not None
                else max(per_test_timeout * 2, per_test_timeout + 30)
            )
            slow_outer = max(slow_timeout * 3, slow_timeout + 60)
            slow_parallel = slow_workers > 1 and len(nodeids_slow) > 1
            if nodeids_slow:
                if slow_parallel:
                    results_slow, first_failure_slow = _run_parallel(
                        nodeids_slow,
                        slow_workers,
                        slow_timeout,
                        slow_outer,
                        args.tb,
                        pytest_args,
                        args.disable_plugin_autoload,
                        args.fail_fast,
                        args.faulthandler_timeout,
                        _and("(slow or veryslow)"),
                        args.kexpr,
                    )
                else:
                    results_slow, first_failure_slow = _run_serial(
                        nodeids_slow,
                        slow_timeout,
                        slow_outer,
                        args.tb,
                        pytest_args,
                        args.disable_plugin_autoload,
                        args.fail_fast,
                        args.faulthandler_timeout,
                        _and("(slow or veryslow)"),
                        args.kexpr,
                    )
            else:
                results_slow, first_failure_slow = ([], None)

            # Combine
            results = list(results_fast) + list(results_slow)
            first_failure = first_failure_fast or first_failure_slow
    elif args.full_suite:
        print("🔍 Discovering tests...")
        nodeids = discover_tests(
            args.markers, args.kexpr, pytest_args, args.disable_plugin_autoload
        )
        if not nodeids:
            print("❌ No tests discovered. Check your markers / -k and pytest arguments.")
            return 1
        if args.limit:
            nodeids = nodeids[: args.limit]
        print(f"📋 Discovered {len(nodeids)} tests")
        # Workers
        if args.workers.strip().lower() == "auto":
            workers = _auto_workers()
        else:
            try:
                workers = max(1, int(args.workers))
            except Exception:
                workers = 1
        parallel = workers > 1 and len(nodeids) > 1
        if parallel:
            results, first_failure = _run_parallel(
                nodeids,
                workers,
                per_test_timeout,
                outer_timeout,
                args.tb,
                pytest_args,
                args.disable_plugin_autoload,
                args.fail_fast,
                args.faulthandler_timeout,
                args.markers,
                args.kexpr,
            )
        else:
            results, first_failure = _run_serial(
                nodeids,
                per_test_timeout,
                outer_timeout,
                args.tb,
                pytest_args,
                args.disable_plugin_autoload,
                args.fail_fast,
                args.faulthandler_timeout,
                args.markers,
                args.kexpr,
            )
    else:
        nodeids = args.nodeids or DEFAULT_NODEIDS
        # Keep only plausible nodeids (avoid stray tokens)
        nodeids = [
            nid for nid in nodeids if ("::" in nid) or nid.endswith(".py") or Path(nid).is_dir()
        ]
        print(f"📋 Running {len(nodeids)} targeted tests")
        print(f"🐍 Using interpreter: {sys.executable}")
        print(f"⏱️  Per-test timeout: {per_test_timeout}s  |  Outer timeout: {outer_timeout}s")
        if args.markers:
            print(f"🏷️  -m: {args.markers}")
        if args.kexpr:
            print(f"🔎  -k: {args.kexpr}")
        if pytest_args:
            print(f"⚙️  Pytest args: {pytest_args}")
        if args.fail_fast:
            print("🚨 FAIL-FAST MODE enabled")
        if args.disable_plugin_autoload:
            print("🧹 Plugin autoload disabled (opt-in only)")
        if args.faulthandler_timeout is not None:
            print(f"🧰 Faulthandler timeout: {args.faulthandler_timeout}s")

        # Workers
        if args.workers.strip().lower() == "auto":
            workers = _auto_workers()
        else:
            try:
                workers = max(1, int(args.workers))
            except Exception:
                workers = 1
        parallel = workers > 1 and len(nodeids) > 1

        # Run
        if parallel:
            results, first_failure = _run_parallel(
                nodeids,
                workers,
                per_test_timeout,
                outer_timeout,
                args.tb,
                pytest_args,
                args.disable_plugin_autoload,
                args.fail_fast,
                args.faulthandler_timeout,
                args.markers,
                args.kexpr,
            )
        else:
            results, first_failure = _run_serial(
                nodeids,
                per_test_timeout,
                outer_timeout,
                args.tb,
                pytest_args,
                args.disable_plugin_autoload,
                args.fail_fast,
                args.faulthandler_timeout,
                args.markers,
                args.kexpr,
            )

    # Summary counts
    counts = {"PASS": 0, "PASS_LINGER": 0, "FAIL": 0, "TIMEOUT": 0, "ERROR": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    print("\n──── Summary ────")
    for k in ("PASS", "PASS_LINGER", "FAIL", "TIMEOUT", "ERROR"):
        if counts.get(k):
            print(f"{k:11}: {counts[k]}")

    if args.fail_fast and first_failure:
        print("\n🚨 FAIL-FAST: First failure details:")
        print(f"   Test    : {first_failure.nodeid}")
        print(f"   Status  : {first_failure.status}")
        print(f"   Duration: {first_failure.duration_s:.2f}s")
        if first_failure.error_details:
            print(f"   Error   : {first_failure.error_details}")

    # Logs
    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path("output")
    text_log = outdir / f"controlled_test_run_{ts}.log"
    summary_log = outdir / f"failure_summary_{ts}.txt"
    json_log = outdir / f"results_{ts}.json"
    write_text_log(results, text_log, args.show_only_failures)
    write_failure_summary(results, summary_log)
    write_json(results, json_log)

    print(f"\n📋 Detailed log : {text_log}")
    print(f"🚨 Failure summary: {summary_log}")
    print(f"🗂️  JSON results  : {json_log}")

    # Exit codes (preserve your semantics)
    exit_code = 0
    if counts["TIMEOUT"]:
        exit_code = 3
    elif counts["FAIL"]:
        exit_code = 2
    elif counts["ERROR"]:
        exit_code = 4

    if exit_code == 0:
        print("\n🎉 All tests completed successfully!")
    else:
        print(f"\n⚠️  Tests completed with issues (exit code: {exit_code})")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
