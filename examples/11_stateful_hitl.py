"""Example: Stateful correction loop with human input.

This script corresponds to `docs/cookbook/hitl_stateful_correction_loop.md`.
"""

from __future__ import annotations

import asyncio
from flujo import Flujo, Step, Pipeline
from flujo.testing.utils import StubAgent


def build_runner() -> Flujo[str, str]:
    loop_body = Step("draft", StubAgent(["bad", "good"])) >> Step.human_in_the_loop("fix")
    loop = Step.loop_until(
        name="correction",
        loop_body_pipeline=Pipeline.from_step(loop_body),
        exit_condition_callable=lambda out, _ctx: out == "ok",
        max_loops=2,
    )
    return Flujo(loop)


async def main() -> None:
    runner = build_runner()
    print("🚀 Starting correction loop...")
    result = None
    async for item in runner.run_async("start"):
        result = item
    ctx = result.final_pipeline_context
    print(f"Paused with message: {ctx.scratchpad['pause_message']}")

    responses = iter(["not ok", "ok"])
    while ctx.scratchpad.get("status") == "paused":
        response = next(responses)
        print(f"\nSimulated human responds: {response}")
        result = await runner.resume_async(result, response)
        ctx = result.final_pipeline_context
        if ctx.scratchpad.get("status") == "paused":
            print(f"Paused again with message: {ctx.scratchpad['pause_message']}")

    print("\n✅ Pipeline finished!")
    print("Final output:", result.step_history[-1].output)


if __name__ == "__main__":
    asyncio.run(main())
