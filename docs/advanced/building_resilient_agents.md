# Building Resilient Agents

Flujo provides monitoring capabilities that make it simple to add observability to your agents.
These helpers are not only for complex workflows with fallbacks. **Every** agent
benefits from visibility into its execution. By standardizing these concerns at the
framework level you get performance insights without extra boilerplate.

## Input and Output Validation

For input and output validation, use Flujo's validation system with `Step.validate()` instead of
agent-level decorators. This provides more flexibility and follows Flujo's policy-driven architecture:

```python
from pydantic import BaseModel
from flujo.domain.validation import BaseValidator, ValidationResult
from flujo.domain import Step

class InputValidator(BaseValidator):
    async def validate(self, data, *, context=None) -> ValidationResult:
        if not isinstance(data, dict) or 'value' not in data:
            return ValidationResult(
                is_valid=False,
                feedback="Input must be a dict with 'value' key",
                validator_name="InputValidator"
            )
        return ValidationResult(is_valid=True, validator_name="InputValidator")

# Use in your pipeline
pipeline = (
    Step.validate(validators=[InputValidator()])
    >> Step.solution(agent)
    >> Step.validate(validators=[output_validator])
)
```

This approach provides better separation of concerns and more granular control.

## Monitoring Agent Calls

`@monitored_agent` records execution metrics using the global monitor instance.

```python
from flujo.agents import monitored_agent
from flujo.infra.monitor import global_monitor

@monitored_agent("my_agent")
class MyMonitoredAgent(AsyncAgentProtocol[str, str]):
    async def run(self, data: str, **kwargs) -> str:
        return data.upper()

# After running
# global_monitor.calls contains details of each invocation
```

## Agent Monitoring

Use `@monitored_agent` to capture execution metrics and performance data for your agents:

```python
@monitored_agent("my_agent")
class OptimizedAgent(AsyncAgentProtocol[InputModel, OutputModel]):
    async def run(self, data: InputModel, **kwargs) -> OutputModel:
        return OutputModel(doubled=data.value * 2)
```

For input/output validation, use the Step-level validation system which provides better architectural separation and follows Flujo's policy-driven design.
