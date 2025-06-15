import pytest

from pydantic_ai_orchestrator.domain import Step
from pydantic_ai_orchestrator.application.pipeline_runner import PipelineRunner
from pydantic_ai_orchestrator.plugins.sql_validator import SQLSyntaxValidator
from pydantic_ai_orchestrator.testing.utils import StubAgent


@pytest.mark.e2e
async def test_sql_pipeline_with_real_validator():
    sql_agent = StubAgent(["SELECT FROM"])  # invalid SQL
    validator_agent = StubAgent([None])
    solution_step = Step.solution(sql_agent)
    validation_step = Step.validate_step(validator_agent).add_plugin(SQLSyntaxValidator())
    pipeline = solution_step >> validation_step
    runner = PipelineRunner(pipeline)
    result = await runner.run_async("prompt")
    assert result.step_history[-1].success is False
    assert result.step_history[-1].feedback
