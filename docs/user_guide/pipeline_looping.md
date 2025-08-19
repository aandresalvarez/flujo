# LoopStep: Iterative Pipelines

`LoopStep` allows you to run a sub‑pipeline repeatedly until a custom condition is met. This enables iterative refinement, polling style workflows or self‑correction loops.

## Parameters

- **`name`** – Step name.
- **`loop_body_pipeline`** – A `Pipeline` executed on each iteration.
- **`exit_condition_callable`** – Callable accepting `(last_body_output, context)` and returning `True` to stop looping.
- **`max_loops`** – Maximum iterations (defaults to `5`). Prevents infinite loops.
- **`initial_input_to_loop_body_mapper`** – Optional function mapping the `LoopStep` input to the first body input.
- **`iteration_input_mapper`** – Optional function mapping the previous iteration output to the next iteration input.
- **`loop_output_mapper`** – Optional function mapping the final successful output of the body to the overall `LoopStep` output.

All mappers receive the shared typed pipeline context when provided.

## Using Typed Pipeline Context

A single context instance is passed into every iteration. The loop body can modify this context and the exit condition can inspect it.

```python
from flujo.domain import Step, Pipeline
from flujo.domain.models import PipelineContext

class Ctx(PipelineContext):
    counter: int = 0

async def inc(x: int, *, context: Ctx | None = None) -> int:
    if context:
        context.counter += 1
    return x + 1

body = Pipeline.from_step(Step("inc", inc))

def should_exit(last: int, ctx: Ctx | None) -> bool:
    return last >= 3

loop_step = Step.loop_until(
    name="increment_until_three",
    loop_body_pipeline=body,
    exit_condition_callable=should_exit,
    max_loops=5,
)
```

## Success and Failure

- `StepResult.attempts` reflects the number of iterations performed.
- `success` is `True` when the exit condition is met and the final iteration completed successfully.
- Reaching `max_loops` without meeting the exit condition marks the step as failed.

## Examples

```python
pipeline = Step.solution(agent_a) >> loop_step
runner = Flujo(pipeline, context_model=Ctx)
result = runner.run(0)
print(result.step_history[-1].output)
print(result.final_pipeline_context.counter)
```

See [pipeline_dsl.md](pipeline_dsl.md) for general DSL usage. For a complete script demonstrating `LoopStep`, check [this script on GitHub](https://github.com/aandresalvarez/flujo/blob/main/examples/07_loop_step.py).
