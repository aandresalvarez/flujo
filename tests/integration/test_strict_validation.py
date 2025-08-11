import pytest

from flujo.domain import Step
from flujo.domain.validation import BaseValidator
from flujo.domain.validation import ValidationResult
from flujo.testing.utils import StubAgent, gather_result
from tests.conftest import create_test_flujo


class FailValidator(BaseValidator):
    async def validate(self, output_to_check: str, *, context=None) -> ValidationResult:
        return ValidationResult(is_valid=False, feedback="bad", validator_name=self.name)


@pytest.mark.asyncio
async def test_non_strict_validation_pass_through() -> None:
    agent = StubAgent(["ok"])
    step = Step.validate_step(agent, validators=[FailValidator()], strict=False)
    runner = create_test_flujo(step)
    result = await gather_result(runner, "in")
    hist = result.step_history[0]
    assert hist.success is True
    assert hist.output == "ok"
    assert hist.metadata_ and hist.metadata_["validation_passed"] is False


@pytest.mark.asyncio
async def test_strict_validation_drops_output() -> None:
    agent = StubAgent(["bad"])
    step = Step.validate_step(agent, validators=[FailValidator()], strict=True)
    runner = create_test_flujo(step)
    result = await gather_result(runner, "in")
    hist = result.step_history[0]
    assert hist.success is False
    assert hist.output is None


@pytest.mark.asyncio
async def test_regular_step_keeps_output_on_validation_failure() -> None:
    agent = StubAgent(["value"])
    step = Step.model_validate({"name": "regular", "agent": agent, "validators": [FailValidator()]})
    runner = create_test_flujo(step)
    result = await gather_result(runner, "in")
    hist = result.step_history[0]
    assert hist.success is False
    assert hist.output == "value"
