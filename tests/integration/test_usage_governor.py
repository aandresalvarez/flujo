# mypy: ignore-errors
import pytest
from pydantic import BaseModel

from flujo import Flujo, Step, Pipeline, UsageLimits, UsageLimitExceededError
from flujo.testing.utils import gather_result
from typing import Any
from flujo.domain.agent_protocol import AsyncAgentProtocol
from flujo.domain.models import PipelineResult


class MockAgentOutput(BaseModel):
    """A mock agent output that includes cost and token metrics."""

    value: int
    cost_usd: float = 0.1
    token_counts: int = 100


class FixedMetricAgent(AsyncAgentProtocol[int, MockAgentOutput]):
    """An agent that returns a fixed cost and token count on each call."""

    async def run(self, data: int | MockAgentOutput, **kwargs: Any) -> MockAgentOutput:
        val = data.value if isinstance(data, MockAgentOutput) else data
        return MockAgentOutput(value=val + 1)


@pytest.fixture
def metric_pipeline() -> Pipeline[int, MockAgentOutput]:
    """Provides a simple pipeline with one step that incurs usage."""
    return Pipeline.from_step(Step("metric_step", FixedMetricAgent()))


@pytest.mark.asyncio
async def test_governor_halts_on_cost_limit_breach(
    metric_pipeline: Pipeline[int, MockAgentOutput],
) -> None:
    """Verify pipeline stops when cost limit is exceeded."""
    limits = UsageLimits(total_cost_usd_limit=0.15, total_tokens_limit=None)
    pipeline: Pipeline[int, MockAgentOutput] = metric_pipeline >> Step(
        "second_step",
        FixedMetricAgent(),
    )  # type: ignore[arg-type]
    runner = Flujo(pipeline, usage_limits=limits)

    with pytest.raises(UsageLimitExceededError) as exc_info:
        await gather_result(runner, 0)

    assert "Cost limit of $0.15 exceeded" in str(exc_info.value)
    result: PipelineResult = exc_info.value.result
    assert len(result.step_history) == 2
    assert result.total_cost_usd == 0.2


@pytest.mark.asyncio
async def test_governor_halts_on_token_limit_breach(
    metric_pipeline: Pipeline[int, MockAgentOutput],
) -> None:
    """Verify pipeline stops when token limit is exceeded."""
    limits = UsageLimits(total_cost_usd_limit=None, total_tokens_limit=150)
    pipeline: Pipeline[int, MockAgentOutput] = metric_pipeline >> Step(
        "second_step",
        FixedMetricAgent(),
    )  # type: ignore[arg-type]
    runner = Flujo(pipeline, usage_limits=limits)

    with pytest.raises(UsageLimitExceededError) as exc_info:
        await gather_result(runner, 0)

    assert "Token limit of 150 exceeded" in str(exc_info.value)
    result: PipelineResult = exc_info.value.result
    assert len(result.step_history) == 2
    assert result.step_history[1].token_counts == 100


@pytest.mark.asyncio
async def test_governor_allows_completion_within_limits(
    metric_pipeline: Pipeline[int, MockAgentOutput],
) -> None:
    """Verify pipeline completes when usage is within limits."""
    limits = UsageLimits(total_cost_usd_limit=0.2, total_tokens_limit=200)
    pipeline: Pipeline[int, MockAgentOutput] = metric_pipeline >> Step(
        "second_step",
        FixedMetricAgent(),
    )  # type: ignore[arg-type]
    runner = Flujo(pipeline, usage_limits=limits)

    result = await gather_result(runner, 0)

    assert len(result.step_history) == 2
    assert result.step_history[-1].success
    assert result.total_cost_usd == 0.2


@pytest.mark.asyncio
async def test_governor_inactive_when_no_limits_provided(
    metric_pipeline: Pipeline[int, MockAgentOutput],
) -> None:
    """Verify pipeline runs normally when no limits are set."""
    pipeline: Pipeline[int, MockAgentOutput] = metric_pipeline >> Step(
        "second_step",
        FixedMetricAgent(),
    )  # type: ignore[arg-type]
    runner = Flujo(pipeline)
    result = await gather_result(runner, 0)

    assert len(result.step_history) == 2
    assert result.step_history[-1].success


@pytest.mark.asyncio
async def test_governor_halts_immediately_on_zero_limit(
    metric_pipeline: Pipeline[int, MockAgentOutput],
) -> None:
    """Verify a zero limit halts the pipeline after the first incurring step."""
    limits = UsageLimits(total_cost_usd_limit=0.0, total_tokens_limit=None)
    runner = Flujo(metric_pipeline, usage_limits=limits)

    with pytest.raises(UsageLimitExceededError):
        await gather_result(runner, 0)


@pytest.mark.asyncio
async def test_governor_with_loop_step(
    metric_pipeline: Pipeline[int, MockAgentOutput],
) -> None:
    """Verify the governor works correctly with iterative steps like LoopStep."""
    limits = UsageLimits(total_cost_usd_limit=0.25, total_tokens_limit=None)
    loop_step = Step.loop_until(
        name="governed_loop",
        loop_body_pipeline=metric_pipeline,
        exit_condition_callable=lambda out, ctx: out.value >= 4,
        max_loops=5,
    )
    runner = Flujo(loop_step, usage_limits=limits)

    with pytest.raises(UsageLimitExceededError) as exc_info:
        await gather_result(runner, 0)

    result: PipelineResult = exc_info.value.result
    loop_result = result.step_history[0]
    assert loop_result.attempts == 3
    assert result.total_cost_usd == pytest.approx(0.30)


@pytest.mark.asyncio
async def test_governor_halts_loop_step_mid_iteration(
    metric_pipeline: Pipeline[int, MockAgentOutput],
) -> None:
    """Governor stops a LoopStep when limits are breached mid-loop."""
    limits = UsageLimits(total_cost_usd_limit=0.25, total_tokens_limit=None)
    loop_step = Step.loop_until(
        name="breach_loop",
        loop_body_pipeline=metric_pipeline,
        exit_condition_callable=lambda _out, _ctx: False,
        max_loops=5,
    )
    runner = Flujo(loop_step, usage_limits=limits)

    with pytest.raises(UsageLimitExceededError) as exc_info:
        await gather_result(runner, 0)

    result: PipelineResult = exc_info.value.result
    assert len(result.step_history) == 1
    loop_result = result.step_history[0]
    assert not loop_result.success
    assert loop_result.attempts == 3
    assert result.total_cost_usd == pytest.approx(0.30)
