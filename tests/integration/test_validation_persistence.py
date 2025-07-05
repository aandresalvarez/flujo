import pytest
from flujo.domain.models import BaseModel, Field

from flujo.application.runner import Flujo
from flujo.domain import Step
from flujo.validation import BaseValidator
from flujo.domain.validation import ValidationResult
from flujo.testing.utils import StubAgent, gather_result
from flujo.testing.assertions import assert_validator_failed


class PassValidator(BaseValidator):
    async def validate(
        self, output_to_check: str, *, context: BaseModel | None = None
    ) -> ValidationResult:
        return ValidationResult(is_valid=True, validator_name=self.name)


class FailValidator(BaseValidator):
    async def validate(
        self, output_to_check: str, *, context: BaseModel | None = None
    ) -> ValidationResult:
        return ValidationResult(is_valid=False, feedback="bad output", validator_name=self.name)


class Ctx(BaseModel):
    feedback_history: list[str] = Field(default_factory=list)
    validation_history: list[ValidationResult] = Field(default_factory=list)


@pytest.mark.asyncio
async def test_persist_feedback_and_results() -> None:
    agent = StubAgent(["bad"])
    step = Step.validate_step(
        agent,
        validators=[FailValidator()],
        persist_feedback_to_context="feedback_history",
        persist_validation_results_to="validation_history",
    )
    runner = Flujo(step, context_model=Ctx)
    result = await gather_result(runner, "in")
    ctx = result.final_pipeline_context
    assert ctx.feedback_history and ctx.feedback_history[0] == result.step_history[0].feedback
    assert len(ctx.validation_history) == 1
    vr = ctx.validation_history[0]
    assert vr.validator_name == "FailValidator"
    assert not vr.is_valid
    assert "bad output" in (vr.feedback or "")
    assert_validator_failed(result, "FailValidator", "bad output")


@pytest.mark.asyncio
async def test_persist_results_on_success() -> None:
    agent = StubAgent(["ok"])
    step = Step.validate_step(
        agent,
        validators=[PassValidator()],
        persist_validation_results_to="validation_history",
    )
    runner = Flujo(step, context_model=Ctx)
    result = await gather_result(runner, "in")
    ctx = result.final_pipeline_context
    assert ctx.feedback_history == []
    assert len(ctx.validation_history) == 1
    vr = ctx.validation_history[0]
    assert vr.is_valid
    assert vr.validator_name == "PassValidator"
