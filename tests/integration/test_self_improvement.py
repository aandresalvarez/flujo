import pytest
from pydantic_ai_orchestrator.application.self_improvement import (
    evaluate_and_improve,
    SelfImprovementAgent,
)
from pydantic_ai_orchestrator.application.pipeline_runner import PipelineRunner
from pydantic_ai_orchestrator.application.eval_adapter import run_pipeline_async
from pydantic_ai_orchestrator.domain import Step
from pydantic_ai_orchestrator.testing.utils import StubAgent
from pydantic_evals import Dataset, Case


class DummyAgent:
    async def run(self, prompt: str) -> str:  # pragma: no cover - simple stub
        return (
            '{"suggestions": ['
            '{"target_step_name": "solution", "failure_pattern": "error",'
            ' "suggested_change": "fix", "example_failing_cases": ["c1"],'
            ' "suggested_config_change": null, "suggested_new_test_case": null}'
            ']}'
        )


@pytest.mark.asyncio
async def test_evaluate_and_improve_flow():
    agent = StubAgent(["ok"])
    pipeline = Step.solution(agent)
    runner = PipelineRunner(pipeline)
    dataset = Dataset(cases=[Case(inputs="hi", expected_output="wrong")])
    report = await evaluate_and_improve(
        lambda x: run_pipeline_async(x, runner=runner),
        dataset,
        SelfImprovementAgent(DummyAgent()),
    )
    assert report.suggestions[0].target_step_name == "solution"
