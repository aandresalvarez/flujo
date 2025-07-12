# Cookbook: Parallel Research Agent

This recipe demonstrates how to fan out multiple research tasks in parallel and
merge their findings back into a shared context. It uses the new
`merge_strategy` parameter on `Step.parallel`.

```python
from flujo import Flujo, Step
from flujo.domain import MergeStrategy, PipelineContext


class ResearchCtx(PipelineContext):
    pass


class ResearchAgent:
    def __init__(self, topic: str) -> None:
        self.topic = topic

    async def run(self, data: str, *, context: ResearchCtx | None = None) -> str:
        # Imagine an API call here
        context.scratchpad[self.topic] = f"findings about {self.topic}"
        return f"research_{self.topic}"


branches = {
    "a": Step.model_validate({"name": "a", "agent": ResearchAgent("ai")}),
    "b": Step.model_validate({"name": "b", "agent": ResearchAgent("ml")}),
}

parallel = Step.parallel(
    name="research",
    branches=branches,
    merge_strategy=MergeStrategy.MERGE_SCRATCHPAD,
)

runner = Flujo(parallel, context_model=ResearchCtx)
result = runner.run("start", initial_context_data={"initial_prompt": "goal"})
print(result.final_pipeline_context.scratchpad)
```

Running this pipeline yields a scratchpad dictionary containing the findings
from both branches. If two branches attempt to write the same scratchpad key,
a `ValueError` is raised to avoid accidental overwrites.
