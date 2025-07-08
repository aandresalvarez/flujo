"""Demonstrates a resumable workflow using the SQLiteBackend."""
import asyncio
from pathlib import Path
from flujo import Flujo, PipelineRegistry, step, Step
from flujo.state.backends.sqlite import SQLiteBackend

@step
async def to_upper(text: str) -> str:
    return text.upper()

pipeline = to_upper >> Step.human_in_the_loop("approve", message_for_user="Approve?")

registry = PipelineRegistry()
registry.register(pipeline, "demo", "1.0.0")
backend = SQLiteBackend(Path("workflow_state.db"))

async def main() -> None:
    run_id = "demo-run"
    print("\u25B6\uFE0F Starting workflow...")
    runner = Flujo(
        registry=registry,
        pipeline_name="demo",
        pipeline_version="1.0.0",
        state_backend=backend,
    )
    paused = await runner.run_async("hello", initial_context_data={"run_id": run_id})
    print("Paused message:", paused.final_pipeline_context.scratchpad.get("pause_message"))

    print("\n\u23F9 Restarting and resuming...")
    new_runner = Flujo(
        registry=registry,
        pipeline_name="demo",
        pipeline_version="1.0.0",
        state_backend=backend,
    )
    final = await new_runner.resume_async(paused, "yes")
    print("Final output:", final.step_history[-1].output)

if __name__ == "__main__":
    asyncio.run(main())
