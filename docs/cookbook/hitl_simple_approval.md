# Cookbook: Simple Human Approval

Use a `HumanInTheLoopStep` to pause a pipeline until a person approves the result.

```python
from flujo import Step, Pipeline, Flujo
from flujo.testing.utils import StubAgent

pipeline = Step("draft", StubAgent(["draft text"])) >> Step.human_in_the_loop("approve", message_for_user="Approve the draft?")
runner = Flujo(pipeline)
result = await runner.run_async("start")
# show result.final_pipeline_context.scratchpad["pause_message"] to the user
# then resume
result = await runner.resume_async(result, "yes")
```
