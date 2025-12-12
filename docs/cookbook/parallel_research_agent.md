# Cookbook: Parallel Research Agent

This recipe demonstrates how to fan out multiple research tasks in parallel and
merge their findings back into a shared context. It uses the new
`merge_strategy` parameter on `Step.parallel`.

```python
from flujo import Flujo, Step
from flujo.domain import MergeStrategy, PipelineContext


class ResearchCtx(PipelineContext):
    research_results: dict[str, str] = {}


class ResearchAgent:
    def __init__(self, topic: str) -> None:
        self.topic = topic

    async def run(self, data: str, *, context: ResearchCtx | None = None) -> str:
        # Imagine an API call here
        context.research_results[self.topic] = f"findings about {self.topic}"
        return f"research_{self.topic}"


branches = {
    "a": Step.model_validate({"name": "a", "agent": ResearchAgent("ai")}),
    "b": Step.model_validate({"name": "b", "agent": ResearchAgent("ml")}),
}

parallel = Step.parallel(
    name="research",
    branches=branches,
    merge_strategy=MergeStrategy.CONTEXT_UPDATE,
)

runner = Flujo(parallel, context_model=ResearchCtx)
result = runner.run("start", initial_context_data={"initial_prompt": "goal"})
print(result.final_pipeline_context.research_results)
```

Running this pipeline yields a typed `research_results` dictionary containing the findings
from both branches. If two branches attempt to write the same key, `CONTEXT_UPDATE`
will detect the conflict; use `field_mapping` to disambiguate destinations when needed.
