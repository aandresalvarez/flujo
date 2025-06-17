# Cookbook: Stateful Correction Loop

Combine `LoopStep` with human input to allow bounded multi-turn corrections.

```python
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
