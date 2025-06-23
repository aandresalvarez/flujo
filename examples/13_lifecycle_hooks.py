"""Example: Observing pipeline events with lifecycle hooks.

This script corresponds to `docs/cookbook/lifecycle_hooks.md`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from flujo import Flujo, Step
from flujo.exceptions import PipelineAbortSignal
from flujo.domain.events import HookPayload, OnStepFailurePayload


async def simple_logger_hook(payload: HookPayload) -> None:
    """Print the name of each event."""

    print(f"HOOK FIRED: {payload.event_name}")


async def abort_on_failure_hook(payload: OnStepFailurePayload) -> None:
    """Abort the run if any step fails."""

    step_name = payload.step_result.name
    print(f"HOOK: Step '{step_name}' failed. Aborting the run.")
    raise PipelineAbortSignal("Aborted due to failure in '{step_name}'")


def build_runner() -> Flujo[str, str]:
    failing_step = Step("failing", agent=MagicMock(side_effect=RuntimeError("oops")))
    pipeline = Step("start", agent=MagicMock(return_value="ok")) >> failing_step
    runner = Flujo(pipeline, hooks=[simple_logger_hook, abort_on_failure_hook])
    return runner


def main() -> None:
    runner = build_runner()
    print("🚀 Running pipeline with hooks...")
    result = runner.run("hi")
    print(
        f"\nPipeline finished. It ran {len(result.step_history)} steps before being aborted by the hook."
    )


if __name__ == "__main__":
    main()
