#!/usr/bin/env python3
"""
Profile `ExecutorCore.execute` on a minimal hot-path workload using cProfile.

Examples:
    uv run python scripts/profile_executor_core_hotpath.py --iterations 50 --warmup 10
    uv run python scripts/profile_executor_core_hotpath.py --optimized --no-cache
    uv run python scripts/profile_executor_core_hotpath.py --no-fastpath --iterations 25
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import pstats
from pathlib import Path
from typing import Any, Callable

from flujo.application.core.executor_core import ExecutorCore, OptimizationConfig
from flujo.application.core.estimation import build_default_estimator_factory
from flujo.domain.dsl.step import Step
from flujo.domain.models import PipelineContext


def _build_default_suffix(*, cache: bool, optimized: bool, fastpath: bool) -> str:
    parts = ["executor_core_execute"]
    if not cache:
        parts.append("no_cache")
    if fastpath:
        parts.append("fastpath")
    if optimized:
        parts.append("optimized")
    if len(parts) == 1:
        parts.append("baseline")
    return "_".join(parts)


def _default_paths(*, cache: bool, optimized: bool, fastpath: bool) -> tuple[Path, Path]:
    suffix = _build_default_suffix(cache=cache, optimized=optimized, fastpath=fastpath)
    prof_dir = Path("prof")
    return prof_dir / f"{suffix}.prof", prof_dir / f"{suffix}_stats.txt"


def build_executor(*, enable_cache: bool, optimized: bool) -> ExecutorCore[Any]:
    """Construct an ExecutorCore tuned for profiling."""
    opt_config = OptimizationConfig() if optimized else None
    return ExecutorCore(
        enable_cache=enable_cache,
        optimization_config=opt_config,
        estimator_factory=build_default_estimator_factory(),
    )


def build_step() -> Step[int, int]:
    """Create a simple step that exercises the agent runner without side effects."""

    async def _hotpath_agent(
        value: int,
        scale: int = 1,
    ) -> int:
        # Minimal arithmetic to avoid hiding framework overhead
        return value * scale

    return Step.from_callable(_hotpath_agent, name="hotpath_step")


async def run_workload(
    executor: ExecutorCore[Any],
    step: Step[int, int],
    *,
    iterations: int,
    warmup: int,
    fastpath: bool,
) -> None:
    """Execute the step repeatedly to capture steady-state overhead."""
    payload = 1
    context_factory: Callable[[int], PipelineContext | None]
    resources_factory: Callable[[int], dict[str, Any] | None]

    if fastpath:

        def context_factory(_: int) -> None:
            return None

        def resources_factory(_: int) -> None:
            return None
    else:

        def context_factory(i: int) -> PipelineContext:
            return PipelineContext(scratchpad={"delta": i % 3})

        def resources_factory(i: int) -> dict[str, Any]:
            return {"delta": (i + 1) % 3}

    for i in range(warmup):
        await executor.execute(
            step,
            payload,
            context=context_factory(i),
            resources=resources_factory(i),
        )

    for i in range(iterations):
        await executor.execute(
            step,
            payload,
            context=context_factory(i),
            resources=resources_factory(i),
        )


def profile(args: argparse.Namespace) -> tuple[cProfile.Profile, Path, Path]:
    """Run the workload under cProfile and persist both .prof and text stats."""
    profile_out, stats_out = _default_paths(
        cache=args.cache, optimized=args.optimized, fastpath=args.fastpath
    )
    if args.profile_out:
        profile_out = args.profile_out
    if args.stats_out:
        stats_out = args.stats_out

    profile_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.parent.mkdir(parents=True, exist_ok=True)

    executor = build_executor(enable_cache=args.cache, optimized=args.optimized)
    step = build_step()

    prof = cProfile.Profile()
    prof.enable()
    asyncio.run(
        run_workload(
            executor,
            step,
            iterations=args.iterations,
            warmup=args.warmup,
            fastpath=args.fastpath,
        )
    )
    prof.disable()
    prof.dump_stats(profile_out.as_posix())

    with stats_out.open("w", encoding="utf-8") as f:
        stats = pstats.Stats(prof, stream=f).strip_dirs().sort_stats("cumulative")
        stats.print_stats(args.top)

    return prof, profile_out, stats_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200, help="Profiled iterations.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations (not profiled).")
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="Enable OptimizationConfig for the executor.",
    )
    parser.add_argument(
        "--no-cache",
        dest="cache",
        action="store_false",
        help="Disable executor cache to measure raw execution cost.",
    )
    parser.add_argument(
        "--no-fastpath",
        dest="fastpath",
        action="store_false",
        help="Include context/resources to exercise signature analysis path.",
    )
    parser.add_argument(
        "--profile-out",
        type=Path,
        help="Optional .prof output path (defaults to prof/<suffix>.prof).",
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        help="Optional human-readable stats path (defaults to prof/<suffix>_stats.txt).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of cumulative entries to print in stats file.",
    )
    parser.set_defaults(cache=True, fastpath=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, profile_out, stats_out = profile(args)
    print(f"Saved profile to {profile_out}")
    print(f"Saved stats to   {stats_out}")


if __name__ == "__main__":
    main()
