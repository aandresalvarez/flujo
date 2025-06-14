import pytest
from pydantic_ai_orchestrator.application.eval_adapter import run_pipeline_async
from pydantic_ai_orchestrator.application.pipeline_runner import PipelineRunner
from pydantic_ai_orchestrator.domain import Step
from pydantic_ai_orchestrator.domain.models import PipelineResult
from pydantic_ai_orchestrator.testing.utils import StubAgent
from pydantic_evals import Dataset, Case


@pytest.mark.asyncio
async def test_adapter_returns_pipeline_result():
    agent = StubAgent(["ok"])
    pipeline = Step.solution(agent)
    runner = PipelineRunner(pipeline)
    dataset = Dataset(cases=[Case(inputs="hi", expected_output="ok")])

    report = await dataset.evaluate(lambda x: run_pipeline_async(x, runner=runner))
    assert isinstance(report.cases[0].output, PipelineResult)
