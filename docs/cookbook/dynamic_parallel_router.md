# Cookbook: Dynamic Parallel Router

Use a **Dynamic Parallel Router** when a router agent decides at runtime which branches to execute concurrently. This pattern is useful for multi-step assistants that delegate to specialized sub-pipelines based on user intent.

```python
from flujo import Flujo, Step
from flujo.domain import MergeStrategy, PipelineContext

class Ctx(PipelineContext):
    task: str

async def router_agent(data: str, *, context: Ctx | None = None) -> list[str]:
    if "billing" in data.lower():
        return ["billing"]
    if "support" in data.lower():
        return ["support"]
    return ["billing", "support"]

billing = Step.from_mapper(lambda x: f"billing:{x}")
support = Step.from_mapper(lambda x: f"support:{x}")

router = Step.dynamic_parallel_branch(
    name="router",
    router_agent=router_agent,
    branches={"billing": billing, "support": support},
    merge_strategy=MergeStrategy.CONTEXT_UPDATE,
)

runner = Flujo(router, context_model=Ctx)
result = runner.run("Need billing info")
print(result.step_history[-1].metadata_["executed_branches"])  # ['billing']
```

The step runs only the branches returned by `router_agent` and merges their outputs using the same semantics as `Step.parallel`.
