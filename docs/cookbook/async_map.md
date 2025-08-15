# Cookbook: Async Map over a Context Iterable

`Step.map_over()` lets you run a sub-pipeline for each item in a list stored on the pipeline context.

```python
from flujo import Flujo, Step, Pipeline
from flujo.domain.models import PipelineContext

class Numbers(PipelineContext):
    values: list[int]

body = Pipeline.from_step(Step.from_mapper(lambda x: x * 2, name="double"))
map_step = Step.map_over("map", body, iterable_input="values")

runner = Flujo(map_step, context_model=Numbers)
result = runner.run(None, initial_context_data={"values": [1, 2, 3]})
print(result.step_history[-1].output)  # [2, 4, 6]
```

The mapping happens asynchronously when the body pipeline's steps do not share state.
