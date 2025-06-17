# Extending flujo

## Adding a Custom Agent

```python
from pydantic_ai import Agent
class MyAgent(Agent):
    ...
```

## Adding a Reflection Step

The simplified orchestrator no longer performs reflection automatically. To
incorporate strategic feedback, build a custom pipeline using `Step`:

```python
from flujo import Step, Flujo, review_agent, solution_agent, validator_agent, get_reflection_agent

reflection_agent = get_reflection_agent(model="anthropic:claude-3-haiku")

pipeline = (
    Step.review(review_agent)
    >> Step.solution(solution_agent)
    >> Step.validate(validator_agent)
    >> Step.validate(reflection_agent)
)

result = Flujo(pipeline).run("Write a poem")
```

### Creating Custom Step Factories with Pre-configured Plugins

If you frequently use a step with the same set of plugins, you can create your own factory function:

```python
from flujo import Step
from my_app.plugins import MyCustomValidator

def ReusableSQLStep(agent, **config) -> Step:
    '''A solution step that always includes MyCustomValidator.'''
    step = Step.solution(agent, **config)
    step.add_plugin(MyCustomValidator(), priority=10)
    return step

# Usage:
pipeline = ReusableSQLStep(my_sql_agent) >> Step.validate(...)
```

### Creating a Custom Execution Backend

Execution back-ends allow you to control how and where pipeline steps run.
Implement the `ExecutionBackend` protocol and pass your implementation to
`Flujo`.

```python
from flujo.domain.backends import ExecutionBackend, StepExecutionRequest
from flujo.domain.models import StepResult

class LoggingBackend(ExecutionBackend):
    def __init__(self, registry: dict[str, Any]):
        self.agent_registry = registry

    async def execute_step(self, request: StepExecutionRequest) -> StepResult:
        print(f"Executing {request.step.name}")
        agent = request.step.agent
        output = await agent.run(request.input_data)
        return StepResult(name=request.step.name, output=output)

custom_backend = LoggingBackend({})
runner = Flujo(pipeline, backend=custom_backend)
```

For remote back-ends, use the `agent_registry` to safely map agent names to
trusted objects.
