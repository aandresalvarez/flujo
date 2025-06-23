# Cookbook: Stateful Correction Loop

Combine `LoopStep` with human input to allow bounded multi-turn corrections.

```python
from flujo import Step, Pipeline, Flujo
from flujo.testing.utils import StubAgent

loop_body = Step("draft", StubAgent(["bad", "good"])) >> Step.human_in_the_loop("fix")
loop = Step.loop_until(
    name="correction",
    loop_body_pipeline=Pipeline.from_step(loop_body),
    exit_condition_callable=lambda out, ctx: out == "ok",
    max_loops=2,
)
runner = Flujo(loop)
paused = await runner.run_async("start")
paused = await runner.resume_async(paused, "not ok")
final = await runner.resume_async(paused, "ok")
```

A full, runnable version of this example can be found in [examples/11_stateful_hitl.py](https://github.com/aandresalvarez/flujo/blob/main/examples/11_stateful_hitl.py).

