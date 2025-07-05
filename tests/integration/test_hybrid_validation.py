from typing import Any

import pytest
from pydantic import BaseModel

from flujo.application.runner import Flujo
from flujo.domain import Step
from flujo.domain.plugins import PluginOutcome
from flujo.testing.utils import StubAgent, DummyPlugin, gather_result
from flujo.validation import BaseValidator
from flujo.domain.validation import ValidationResult


class PassValidator(BaseValidator):
    async def validate(
        self, output_to_check: Any, *, context: BaseModel | None = None
    ) -> ValidationResult:
        return ValidationResult(is_valid=True, validator_name=self.name)


class FailValidator(BaseValidator):
    async def validate(
        self, output_to_check: Any, *, context: BaseModel | None = None
    ) -> ValidationResult:
        return ValidationResult(is_valid=False, feedback="bad output", validator_name=self.name)


class ContextValidator(BaseValidator):
    async def validate(
        self, output_to_check: Any, *, context: BaseModel | None = None
    ) -> ValidationResult:
        flag = getattr(context, "flag", False) if context else False
        return ValidationResult(
            is_valid=flag, feedback=None if flag else "flag not set", validator_name=self.name
        )


@pytest.mark.asyncio
async def test_successful_hybrid_validation() -> None:
    agent = StubAgent(["ok"])
    step = Step.validate_step(agent, validators=[PassValidator()])
    runner = Flujo(step)
    result = await gather_result(runner, "in")
    assert result.step_history[0].success is True


@pytest.mark.asyncio
async def test_programmatic_check_failure() -> None:
    agent = StubAgent(["bad"])
    step = Step.validate_step(agent, validators=[FailValidator()])
    runner = Flujo(step)
    result = await gather_result(runner, "in")
    history = result.step_history[0]
    assert history.success is False
    assert "FailValidator" in history.feedback
    assert "bad output" in history.feedback


class Ctx(BaseModel):
    flag: bool = False


@pytest.mark.asyncio
async def test_context_aware_validator() -> None:
    agent = StubAgent(["anything"])
    step = Step.validate_step(agent, validators=[ContextValidator()])
    runner = Flujo(step, context_model=Ctx, initial_context_data={"flag": True})
    result = await gather_result(runner, "in")
    assert result.step_history[0].success is True


@pytest.mark.asyncio
async def test_aggregated_feedback() -> None:
    agent = StubAgent(["bad"])
    plugin = DummyPlugin([PluginOutcome(success=False, feedback="plugin fail")])
    step = Step.validate_step(agent, plugins=[(plugin, 0)], validators=[FailValidator()])
    runner = Flujo(step)
    result = await gather_result(runner, "in")
    fb = result.step_history[0].feedback
    assert fb is not None
    assert "plugin fail" in fb
    assert "FailValidator" in fb


@pytest.mark.asyncio
async def test_backward_compatibility_no_validators() -> None:
    agent = StubAgent(["ok"])
    step = Step.validate_step(agent)
    runner = Flujo(step)
    result = await gather_result(runner, "in")
    assert result.step_history[0].success is True
