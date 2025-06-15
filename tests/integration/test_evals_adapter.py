import pytest
from flujo.application.eval_adapter import run_pipeline_async
from flujo.application.flujo_engine import Flujo
from flujo.domain import Step
from flujo.domain.models import PipelineResult
from flujo.testing.utils import StubAgent
from pydantic_evals import Dataset, Case


@pytest.mark.asyncio
async def test_adapter_returns_pipeline_result():
    agent = StubAgent(["ok"])
    pipeline = Step.solution(agent)
    runner = Flujo(pipeline)
    dataset = Dataset(cases=[Case(inputs="hi", expected_output="ok")])

    report = await dataset.evaluate(lambda x: run_pipeline_async(x, runner=runner))
    assert isinstance(report.cases[0].output, PipelineResult)
