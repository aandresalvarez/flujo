# Cookbook: Structured Human Input

Validate human input against a Pydantic model for robustness.

```python
from pydantic import BaseModel
from flujo import Step, Flujo
from flujo.testing.utils import StubAgent

class Answer(BaseModel):
    choice: int

step = Step.human_in_the_loop("pick", input_schema=Answer)
pipeline = Step("start", StubAgent(["Q"])) >> step
runner = Flujo(pipeline)
paused = await runner.run_async("x")
# paused.final_pipeline_context.scratchpad["pause_message"] has the question
resumed = await runner.resume_async(paused, {"choice": 1})
assert isinstance(resumed.step_history[-1].output, Answer)
```
