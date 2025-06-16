# Controlling LLM Costs

The Usage Governor lets you define limits for total cost and token usage.
When a pipeline exceeds either limit it stops and raises
`UsageLimitExceededError`.

```python
from flujo import Flujo, Step, UsageLimits, UsageLimitExceededError

class CheapAgent:
    async def run(self, x: int) -> int:
        class Out(BaseModel):
            value: int
            cost_usd: float = 0.05
            token_counts: int = 50
        return Out(value=x + 1)

pipeline = Step("cheap", CheapAgent())
limits = UsageLimits(total_cost_usd_limit=0.1)
runner = Flujo(pipeline, usage_limits=limits)

try:
    runner.run(0)
except UsageLimitExceededError as e:
    print("Stopped early", e)
```
