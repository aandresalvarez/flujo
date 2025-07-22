import pytest
import functools
from flujo.application.eval_adapter import run_pipeline_async
from flujo.domain import Step
from flujo.domain.models import PipelineResult
from flujo.testing.utils import StubAgent
from tests.conftest import create_test_flujo
from pydantic_evals import Dataset, Case


@pytest.mark.asyncio
async def test_adapter_returns_pipeline_result():
    agent = StubAgent(["ok"])
    pipeline = Step.solution(agent)
    runner = create_test_flujo(pipeline)
    dataset = Dataset(cases=[Case(inputs="hi", expected_output="ok")])

    report = await dataset.evaluate(functools.partial(run_pipeline_async, runner=runner))
    assert isinstance(report.cases[0].output, PipelineResult)
