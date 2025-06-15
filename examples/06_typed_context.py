"""
06_typed_context.py
-------------------
Demonstrates sharing state across steps with Typed Pipeline Context.
"""

from pydantic import BaseModel
from flujo import Step, Flujo


class Ctx(BaseModel):
    count: int = 0


async def increment(data: str, *, pipeline_context: Ctx | None = None) -> str:
    if pipeline_context:
        pipeline_context.count += 1
    return data + "!"


pipeline = Step("first", increment) >> Step("second", increment)
runner = Flujo(pipeline, context_model=Ctx)

result = runner.run("hello")
print("Final output:", result.step_history[-1].output)
print("Final count:", result.final_pipeline_context.count)

