# Cookbook: Stateful Correction Loop

Combine `LoopStep` with human input to allow bounded multi-turn corrections.

```python
from flujo import Step, Pipeline, Flujo
from flujo.testing.utils import StubAgent

loop_body = Step("draft", StubAgent(["Initial draft: Hello world", "Improved draft: Hello, world!"])) >> Step.human_in_the_loop("fix")
loop = Step.loop_until(
    name="correction",
    loop_body_pipeline=Pipeline.from_step(loop_body),
    exit_condition_callable=lambda out, ctx: out == "APPROVED",
    max_loops=2,
)
runner = Flujo(loop)
result = None
async for item in runner.run_async("start"):
    result = item
paused = result
paused = await runner.resume_async(paused, "NEEDS_REVISION")
final = await runner.resume_async(paused, "APPROVED")
```
