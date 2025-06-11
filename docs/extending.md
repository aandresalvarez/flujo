# Extending pydantic-ai-orchestrator

## Adding a Custom Agent

```python
from pydantic_ai_orchestrator.infra.agents import Agent
class MyAgent(Agent):
    ...
```

## Custom Scoring

```python
from pydantic_ai_orchestrator.domain.scoring import weighted_score
# Use your own weights
def my_score(checklist, passed):
    weights = {"item1": 0.7, "item2": 0.3}
    return weighted_score(checklist, weights, passed)
```

## Custom Settings

```python
from pydantic_ai_orchestrator.infra.settings import settings
print(settings.max_iters)
``` 