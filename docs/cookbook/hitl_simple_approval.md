# Cookbook: Simple Human Approval

Use a `HumanInTheLoopStep` to pause a pipeline until a person approves the result.

```python
from flujo import Step, Pipeline, Flujo
from flujo.testing.utils import StubAgent

pipeline = Step("draft", StubAgent(["Draft email: Please review the quarterly report for Q3 2024."])) >> Step.human_in_the_loop("approve", message_for_user="Approve the draft?")
# "run_async" returns an async iterator, so consume it to get the result
runner = Flujo(pipeline)
result = None
async for item in runner.run_async("start"):
    result = item
# show result.final_pipeline_context.pause_message to the user
# then resume
result = await runner.resume_async(result, "yes")
```
