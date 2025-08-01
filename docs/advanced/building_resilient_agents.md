# Building Resilient Agents

Flujo provides decorators that make it simple to add validation and monitoring to your agents.
These helpers are not only for complex workflows with fallbacks. **Every** agent
benefits from validated inputs/outputs and visibility into its execution. By
standardizing these concerns at the framework level you get data integrity and
performance insights without extra boilerplate.

## Validating Inputs and Outputs

Use `@validated_agent` to ensure your agent's `run` method receives and returns data that matches your Pydantic models.

```python
from pydantic import BaseModel
from flujo.agents import validated_agent
from flujo.domain.agent_protocol import AsyncAgentProtocol

class InputModel(BaseModel):
    value: int

class OutputModel(BaseModel):
    doubled: int

@validated_agent(InputModel, OutputModel)
class MyAgent(AsyncAgentProtocol[InputModel, OutputModel]):
    async def run(self, data: InputModel, **kwargs) -> OutputModel:
        return OutputModel(doubled=data.value * 2)
```

Passing invalid data will raise `AgentIOValidationError`.

## Monitoring Agent Calls

`@monitored_agent` records execution metrics using the global monitor instance.

```python
from flujo.agents import monitored_agent
from flujo.monitor import global_monitor

@monitored_agent("my_agent")
class MyMonitoredAgent(AsyncAgentProtocol[str, str]):
    async def run(self, data: str, **kwargs) -> str:
        return data.upper()

# After running
# global_monitor.calls contains details of each invocation
```

## Combining Decorators

Decorators are composable. Apply `@monitored_agent` on top so monitoring captures validation failures.

```python
@monitored_agent("combo")
@validated_agent(InputModel, OutputModel)
class CombinedAgent(AsyncAgentProtocol[InputModel, OutputModel]):
    async def run(self, data: InputModel, **kwargs) -> OutputModel:
        return OutputModel(doubled=data.value * 2)
```
