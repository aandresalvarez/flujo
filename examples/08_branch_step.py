"""
08_branch_step.py
-----------------
Demonstrates ConditionalStep for routing to different pipelines.
"""

from flujo import Step, Pipeline, PipelineRunner


def classify(text: str) -> str:
    return "shout" if text.endswith("!") else "normal"


shout_pipeline = Pipeline.from_step(Step("shout", lambda x: x.upper()))
normal_pipeline = Pipeline.from_step(Step("normal", lambda x: x.capitalize()))

branch_step = Step.branch_on(
    name="router",
    condition_callable=lambda out, ctx: out,
    branches={"shout": shout_pipeline, "normal": normal_pipeline},
)

pipeline = Step("classify", classify) >> branch_step
runner = PipelineRunner(pipeline)

result = runner.run("hello!")
print("Final output:", result.step_history[-1].output)

