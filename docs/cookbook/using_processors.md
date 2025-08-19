# Cookbook: Using Processors

`Processor` objects let you modify step inputs or outputs without changing your agents. They are useful for injecting context variables or cleaning up responses.

## Basic Example

```python
from flujo import Step, AgentProcessors
from flujo.processors import AddContextVariables, StripMarkdownFences
from flujo.testing.utils import StubAgent
from flujo.domain.models import PipelineContext

class Ctx(PipelineContext):
    product: str

agent = StubAgent(["```text\nHello!\n```"])
processors = AgentProcessors(
    prompt_processors=[AddContextVariables(["product"])],
    output_processors=[StripMarkdownFences("text")],
)
step = Step.solution(agent, processors=processors)
runner = Flujo(step, context_model=Ctx, initial_context_data={"product": "Widget"})
result = runner.run("Write a slogan")
print(result.step_history[0].output)
```

This prints `Hello!` because the output processor stripped the markdown fences.

## Custom Processor

Create your own processor by implementing the `Processor` protocol:

```python
from flujo.processors import Processor

class Exclaim(Processor):
    name = "Exclaim"

    async def process(self, data: str, context=None) -> str:
        return data + "!"
```

Attach it to a step via `AgentProcessors`.

## Parsing JSON Outputs

Use `EnforceJsonResponse` to automatically convert string replies into Python
objects.

```python
from flujo import Step, Flujo, AgentProcessors
from flujo.processors import EnforceJsonResponse
from flujo.testing.utils import StubAgent

agent = StubAgent(['{"count": 1, "status": "success"}'])
processors = AgentProcessors(output_processors=[EnforceJsonResponse()])
step = Step.solution(agent, processors=processors)
runner = Flujo(step)
result = runner.run("Give me a JSON object")
print(result.step_history[0].output)
```

This prints `{'count': 1, 'status': 'success'}` because the processor parsed the JSON string.

## Serializing Pydantic Models

`SerializePydantic` turns any object with a `model_dump()` method into a plain dictionary. Use it when a downstream agent doesn't understand Pydantic models.

See [Serialize Pydantic Models](serialize_pydantic.md) for a complete example.
