#!/usr/bin/env python3
"""
Compare two cProfile outputs and highlight cumulative-time differences.

Usage:
    python scripts/compare_profiles.py --a prof/executor_core_execute_no_cache.prof \
        --b prof/executor_core_execute_no_cache_optimized.prof --top 30
"""

from __future__ import annotations

import argparse
import pstats
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


FuncKey = Tuple[str, int, str]  # (file, line, func)


@dataclass(frozen=True)
class StatEntry:
    ncalls: int
    tottime: float
    cumtime: float


def load_profile(path: Path) -> Dict[FuncKey, StatEntry]:
    stats = pstats.Stats(str(path)).strip_dirs().sort_stats("cumulative")
    entries: Dict[FuncKey, StatEntry] = {}
    for func, (cc, nc, tt, ct, _) in stats.stats.items():
        # func is (filename, lineno, funcname)
        entries[func] = StatEntry(ncalls=nc, tottime=tt, cumtime=ct)
    return entries


def percent_delta(old: float, new: float) -> float:
    if old == 0.0:
        return float("inf") if new > 0 else 0.0
    return ((new - old) / old) * 100.0


def format_func(func: FuncKey) -> str:
    filename, lineno, funcname = func
    return f"{filename}:{lineno}({funcname})"


def compare_profiles(path_a: Path, path_b: Path, top: int) -> None:
    stats_a = load_profile(path_a)
    stats_b = load_profile(path_b)

    all_funcs = set(stats_a) | set(stats_b)
    merged = []
    for func in all_funcs:
        a = stats_a.get(func, StatEntry(0, 0.0, 0.0))
        b = stats_b.get(func, StatEntry(0, 0.0, 0.0))
        max_cum = max(a.cumtime, b.cumtime)
        merged.append((func, a, b, max_cum))

    merged.sort(key=lambda item: item[3], reverse=True)
    merged = merged[:top]

    header = (
        f"{'cumA(ms)':>10} {'cumB(ms)':>10} {'delta%':>8} {'callsA':>8} {'callsB':>8}  function"
    )
    print(header)
    print("-" * len(header))

    for func, a, b, _ in merged:
        delta = percent_delta(a.cumtime, b.cumtime)
        delta_str = "inf" if delta == float("inf") else f"{delta:>7.1f}"
        print(
            f"{a.cumtime * 1000:10.3f} {b.cumtime * 1000:10.3f} {delta_str:>8} "
            f"{a.ncalls:8d} {b.ncalls:8d}  {format_func(func)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, type=Path, help="Baseline profile path")
    parser.add_argument("--b", required=True, type=Path, help="Comparison profile path")
    parser.add_argument("--top", type=int, default=30, help="Number of functions to display")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_profiles(args.a, args.b, args.top)


if __name__ == "__main__":
    main()
