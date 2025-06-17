"""Demonstrates pausing a pipeline for human input."""
import asyncio
from typing import Any
from flujo import Flujo, Step
from flujo.testing.utils import StubAgent


async def main() -> None:
    pipeline = Step("draft", StubAgent(["A short draft"])) >> Step.human_in_the_loop(
        "approval", message_for_user="Approve draft?"
    )
    runner = Flujo(pipeline)
    result = None
    async for item in runner.run_async("start"):
        result = item
    msg = result.final_pipeline_context.scratchpad.get("pause_message")
    print(f"Pipeline paused with message: {msg}")
    resumed = await runner.resume_async(result, "yes")
    print("Final output:", resumed.step_history[-1].output)


if __name__ == "__main__":
    asyncio.run(main())
