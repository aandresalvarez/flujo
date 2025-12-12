"""Durable workflow example using the SQLiteBackend.

Run once to pause the pipeline and again to resume. The run is identified by a
custom ``run_id`` stored in the context.
"""

import asyncio
from pathlib import Path

from flujo import Flujo, PipelineRegistry, Step, step
from flujo.state import SQLiteBackend
from flujo.testing import StubAgent


@step
async def draft(text: str) -> str:
    print("Generating draft...")
    return "A short draft"


pipeline = (
    Step.solution(StubAgent(["ignored"]))
    >> Step.solution(draft)
    >> Step.human_in_the_loop(name="approval", message_for_user="Approve the draft?")
    >> Step.solution(StubAgent(["Final answer"]))
)

registry = PipelineRegistry()
registry.register(pipeline, "durable_demo", "1.0.0")
backend = SQLiteBackend(Path("workflow_state.db"))


async def main() -> None:
    run_id = "example-run"
    runner = Flujo(
        registry=registry,
        pipeline_name="durable_demo",
        pipeline_version="1.0.0",
        state_backend=backend,
        delete_on_completion=False,
    )

    result = None
    async for item in runner.run_async("start", initial_context_data={"run_id": run_id}):
        result = item
    if result and result.final_pipeline_context.status == "paused":
        print("Pipeline paused. Resuming...")
        runner2 = Flujo(
            registry=registry,
            pipeline_name="durable_demo",
            pipeline_version="1.0.0",
            state_backend=backend,
            delete_on_completion=False,
        )
        resumed = await runner2.resume_async(result, "yes")
        print("Final output:", resumed.step_history[-1].output)
    else:
        print("Run completed without pause.")


if __name__ == "__main__":
    asyncio.run(main())
