import pytest

from flujo.domain import Step
from flujo.plugins.sql_validator import SQLSyntaxValidator
from flujo.testing.utils import StubAgent, gather_result
from tests.conftest import create_test_flujo


@pytest.mark.e2e
async def test_sql_pipeline_with_real_validator():
    sql_agent = StubAgent(["SELECT FROM"])  # invalid SQL
    validator_agent = StubAgent([None])
    solution_step = Step.solution(sql_agent)
    validation_step = Step.validate_step(validator_agent).add_plugin(SQLSyntaxValidator())
    pipeline = solution_step >> validation_step
    runner = create_test_flujo(pipeline)
    result = await gather_result(runner, "prompt")
    assert result.step_history[-1].success is False
    assert result.step_history[-1].feedback
