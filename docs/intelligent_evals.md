# Intelligent Evaluations

This guide explains how to run automated evaluations and use the self-improvement agent introduced in v2.1.

## Quick start

```python
from pydantic_ai_orchestrator.application.eval_adapter import run_pipeline_async
from pydantic_ai_orchestrator.application.self_improvement import evaluate_and_improve, SelfImprovementAgent
from pydantic_ai_orchestrator.application.pipeline_runner import PipelineRunner
from pydantic_ai_orchestrator.domain import Step
from pydantic_evals import Dataset, Case
from pydantic_ai_orchestrator.infra.agents import self_improvement_agent

pipeline = Step.solution(lambda x: x)
runner = PipelineRunner(pipeline)
dataset = Dataset(cases=[Case(inputs="hi", expected_output="hi")])
agent = SelfImprovementAgent(self_improvement_agent)
report = await evaluate_and_improve(
    lambda x: run_pipeline_async(x, runner=runner),
    dataset,
    agent,
)
print(report)
```

The `ImprovementReport` contains structured suggestions for updating your pipeline or evaluation suite.
