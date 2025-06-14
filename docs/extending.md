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
