# Modular Workflows

`flujo` provides factory methods for building reusable workflow components. `Step.from_mapper` wraps simple async functions as steps, while `Step.map_over` and `Step.parallel` compose pipelines for iterative or concurrent execution.

## `Step.from_mapper`

`Step.from_mapper` is a convenience around `Step.from_callable`. It infers input and output types from the annotated async function and creates a `Step`.

```python
async def to_upper(text: str) -> str:
    return text.upper()

upper_step = Step.from_mapper(to_upper)
```

## `Step.map_over`

`Step.map_over` runs a pipeline for each item in an iterable stored in the pipeline context. The collected outputs are returned as a list.

```python
from flujo.domain.models import PipelineContext

class Ctx(PipelineContext):
    nums: list[int]

body = Pipeline.from_step(Step.from_mapper(lambda x: x * 2, name="double"))
mapper = Step.map_over("mapper", body, iterable_input="nums")
runner = Flujo(mapper, context_model=Ctx)
result = runner.run(None, initial_context_data={"nums": [1, 2, 3]})
print(result.step_history[-1].output)  # [2, 4, 6]
```

The mapping pipeline can also run in parallel when its steps are free of side effects.

```python
class SleepAgent:
    async def run(self, data: int) -> int:
        await asyncio.sleep(0.01)
        return data

body = Pipeline.from_step(Step("sleep", SleepAgent()))
mapper = Step.map_over("mapper_par", body, iterable_input="nums")
runner = Flujo(mapper, context_model=Ctx)
result = runner.run(None, initial_context_data={"nums": [0, 1, 2, 3]})
print(result.step_history[-1].output)  # [0, 1, 2, 3]
```

## `Step.parallel`

`Step.parallel` executes multiple branch pipelines concurrently and aggregates their outputs in a dictionary keyed by branch name.

```python
from flujo.domain.models import PipelineContext

class Ctx(PipelineContext):
    val: int = 0

class AddAgent:
    def __init__(self, inc: int) -> None:
        self.inc = inc

    async def run(self, data: int, *, context: Ctx | None = None) -> int:
        if context is not None:
            context.val += self.inc
        await asyncio.sleep(0)
        return data + self.inc

branches = {
    "a": Step("a", AddAgent(1)),
    "b": Step("b", AddAgent(2)),
}
parallel = Step.parallel("par", branches)
runner = Flujo(parallel, context_model=Ctx)
result = runner.run(0)
print(result.step_history[-1].output)  # {"a": 1, "b": 2}
print(result.final_pipeline_context.val)  # 0
```

`Step.parallel` also supports merging branch contexts and flexible failure
handling via the `merge_strategy` and `on_branch_failure` parameters. See
[`pipeline_dsl.md`](pipeline_dsl.md) for details.

See [pipeline_dsl.md](pipeline_dsl.md) for general DSL usage.
