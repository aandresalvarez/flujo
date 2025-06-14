# Extending pydantic-ai-orchestrator

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
from pydantic_ai_orchestrator import Step, PipelineRunner, review_agent, solution_agent, validator_agent, get_reflection_agent

reflection_agent = get_reflection_agent(model="anthropic:claude-3-haiku")

pipeline = (
    Step.review(review_agent)
    >> Step.solution(solution_agent)
    >> Step.validate(validator_agent)
    >> Step.validate(reflection_agent)
)

result = PipelineRunner(pipeline).run("Write a poem")
```

### Creating Custom Step Factories with Pre-configured Plugins

If you frequently use a step with the same set of plugins, you can create your own factory function:

```python
from pydantic_ai_orchestrator import Step
from my_app.plugins import MyCustomValidator

def ReusableSQLStep(agent, **config) -> Step:
    '''A solution step that always includes MyCustomValidator.'''
    step = Step.solution(agent, **config)
    step.add_plugin(MyCustomValidator(), priority=10)
    return step

# Usage:
pipeline = ReusableSQLStep(my_sql_agent) >> Step.validate(...)
```
