# Cookbook: State Machine Pipeline

Use the **state machine pipeline factory** to orchestrate a workflow where the next step to run is determined by context fields. The pipeline loops until a completion flag is set.

```python
from pydantic import BaseModel
from flujo import Flujo, Step, step
from flujo.recipes import make_state_machine_pipeline
from flujo.domain.models import PipelineContext

class Ctx(PipelineContext):
    next_state: str = "start"
    is_complete: bool = False
    counter: int = 0

@step
async def start(data: str, *, context: Ctx) -> str:
    context.counter += 1
    # Transition to "end" after the first iteration
    context.next_state = "end" if context.counter > 1 else "start"
    return data

@step
async def end(data: str, *, context: Ctx) -> str:
    context.is_complete = True
    return f"done after {context.counter} loops"

pipeline = make_state_machine_pipeline(
    nodes={"start": start, "end": end},
    context_model=Ctx,
    router_field="next_state",
    end_state_field="is_complete",
)

runner = Flujo(pipeline, context_model=Ctx)
result = runner.run("go")
print(result.output)
```

This pattern removes boilerplate by handling the loop and state dispatch for you.
