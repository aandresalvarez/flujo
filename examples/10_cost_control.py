"""Example: Enforcing cost limits with `UsageLimits`.

This script corresponds to `docs/cookbook/cost_control.md`.
"""

from __future__ import annotations

from pydantic import BaseModel

from flujo import Flujo, Step, UsageLimits, UsageLimitExceededError


class CostlyAgent:
    """An agent that always incurs a fixed cost."""

    async def run(self, x: int) -> BaseModel:
        class Output(BaseModel):
            value: int
            cost_usd: float = 0.05
            token_counts: int = 50

        return Output(value=x + 1)


def build_runner() -> Flujo[int, BaseModel]:
    pipeline = (
        Step("step_1", CostlyAgent())
        >> Step("step_2", CostlyAgent())
        >> Step("step_3", CostlyAgent())
    )
    limits = UsageLimits(total_cost_usd_limit=0.12)
    return Flujo(pipeline, usage_limits=limits)


def main() -> None:
    runner = build_runner()
    print("🚀 Running pipeline with a cost limit of $0.12...")
    try:
        runner.run(0)
    except UsageLimitExceededError as exc:
        print("\n✅ Pipeline halted as expected!")
        print(f"   Reason: {exc}")
        print(f"   The pipeline ran {len(exc.result.step_history)} steps before stopping.")
        print(f"   Final recorded cost was ${exc.result.total_cost_usd:.2f}")


if __name__ == "__main__":
    main()
