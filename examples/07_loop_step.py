"""
07_loop_step.py
---------------
Demonstrates using LoopStep for iterative refinement.
"""

from pydantic_ai_orchestrator import Step, Pipeline, PipelineRunner


async def add_exclamation(data: str) -> str:
    return data + "!"


body = Pipeline.from_step(Step("add", add_exclamation))

loop_step = Step.loop_until(
    name="refine",
    loop_body_pipeline=body,
    exit_condition_callable=lambda out, ctx: out.endswith("!!!"),
    max_loops=5,
)

runner = PipelineRunner(loop_step)
result = runner.run("hi")
print("Final output:", result.step_history[-1].output)

