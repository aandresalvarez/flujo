import pytest
from flujo.domain.models import BaseModel

from flujo import Flujo, Step, Pipeline
from flujo.domain import UsageLimits
from flujo.exceptions import UsageLimitExceededError
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
    return Pipeline.from_step(
        Step.model_validate({"name": "metric_step", "agent": FixedMetricAgent()})
    )


@pytest.mark.asyncio
async def test_governor_halts_on_cost_limit_breach(
    metric_pipeline: Pipeline[int, MockAgentOutput],
) -> None:
    """Verify pipeline stops when cost limit is exceeded."""
    limits = UsageLimits(total_cost_usd_limit=0.15, total_tokens_limit=None)
    pipeline: Pipeline[int, MockAgentOutput] = metric_pipeline >> Step.model_validate(
        {"name": "s1", "agent": FixedMetricAgent()}
    )
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
    pipeline: Pipeline[int, MockAgentOutput] = metric_pipeline >> Step.model_validate(
        {"name": "s1", "agent": FixedMetricAgent()}
    )
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
    pipeline: Pipeline[int, MockAgentOutput] = metric_pipeline >> Step.model_validate(
        {"name": "s1", "agent": FixedMetricAgent()}
    )
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
    pipeline: Pipeline[int, MockAgentOutput] = metric_pipeline >> Step.model_validate(
        {"name": "s1", "agent": FixedMetricAgent()}
    )
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


@pytest.mark.asyncio
async def test_governor_parallel_step_limit() -> None:
    """Governor stops a ParallelStep when cumulative usage exceeds limits."""
    branches = {
        "a": Step.model_validate({"name": "a", "agent": FixedMetricAgent()}),
        "b": Step.model_validate({"name": "b", "agent": FixedMetricAgent()}),
    }
    parallel = Step.parallel("parallel_usage", branches)
    limits = UsageLimits(total_cost_usd_limit=0.15)
    runner = Flujo(parallel, usage_limits=limits)

    with pytest.raises(UsageLimitExceededError) as exc_info:
        await gather_result(runner, 0)

    assert "Cost limit of $0.15 exceeded" in str(exc_info.value)


class VariableMetricAgent(AsyncAgentProtocol[int, MockAgentOutput]):
    """Agent that allows custom cost and token metrics."""

    def __init__(self, cost: float = 0.1, tokens: int = 100) -> None:
        self.cost = cost
        self.tokens = tokens

    async def run(self, data: int | MockAgentOutput, **kwargs: Any) -> MockAgentOutput:
        val = data.value if isinstance(data, MockAgentOutput) else data
        return MockAgentOutput(value=val + 1, cost_usd=self.cost, token_counts=self.tokens)


@pytest.mark.asyncio
async def test_governor_loop_with_nested_parallel_limit() -> None:
    """Governor enforces limits in LoopStep containing a ParallelStep."""
    branches = {
        "a": Step.model_validate({"name": "a", "agent": VariableMetricAgent(cost=0.1)}),
        "b": Step.model_validate({"name": "b", "agent": VariableMetricAgent(cost=0.1)}),
    }
    parallel = Step.parallel("inner_parallel", branches)
    loop_step = Step.loop_until(
        name="outer_loop",
        loop_body_pipeline=Pipeline.from_step(parallel),
        exit_condition_callable=lambda _out, _ctx: False,
        iteration_input_mapper=lambda _out, _ctx, _i: 0,
        max_loops=10,
    )
    limits = UsageLimits(total_cost_usd_limit=0.5)
    runner = Flujo(loop_step, usage_limits=limits)

    with pytest.raises(UsageLimitExceededError) as exc_info:
        await gather_result(runner, 0)

    assert "Cost limit of $0.5 exceeded" in str(exc_info.value)
    result: PipelineResult = exc_info.value.result
    assert result.total_cost_usd == pytest.approx(0.6)
    assert len(result.step_history) == 1
    loop_result = result.step_history[0]
    assert not loop_result.success
    assert loop_result.attempts == 3
    assert loop_result.cost_usd == pytest.approx(0.6)
    assert "limit" in (loop_result.feedback or "").lower()


@pytest.mark.asyncio
async def test_governor_parallel_limit_first_branch_exceeds() -> None:
    """Governor triggers when the first parallel branch breaches the limit."""
    branches = {
        "expensive": Step.model_validate(
            {"name": "expensive", "agent": VariableMetricAgent(cost=0.6)}
        ),
        "cheap": Step.model_validate({"name": "cheap", "agent": VariableMetricAgent(cost=0.1)}),
    }
    parallel = Step.parallel("parallel_breach", branches)
    loop_step = Step.loop_until(
        name="loop_parallel_breach",
        loop_body_pipeline=Pipeline.from_step(parallel),
        iteration_input_mapper=lambda _out, _ctx, _i: 0,
        exit_condition_callable=lambda _out, _ctx: False,
        max_loops=10,
    )
    limits = UsageLimits(total_cost_usd_limit=0.5)
    runner = Flujo(loop_step, usage_limits=limits)

    with pytest.raises(UsageLimitExceededError):
        await gather_result(runner, 0)


@pytest.mark.asyncio
async def test_governor_cumulative_cost_updates() -> None:
    """Total cost should accumulate correctly across loop iterations."""
    branches = {
        "a": Step.model_validate({"name": "a", "agent": VariableMetricAgent(cost=0.1)}),
        "b": Step.model_validate({"name": "b", "agent": VariableMetricAgent(cost=0.1)}),
    }
    parallel = Step.parallel("count_parallel", branches)

    iteration_counter = 0

    def _exit_after_four(_out: Any, _ctx: Any) -> bool:
        nonlocal iteration_counter
        iteration_counter += 1
        return iteration_counter >= 4

    loop_step = Step.loop_until(
        name="count_loop",
        loop_body_pipeline=Pipeline.from_step(parallel),
        iteration_input_mapper=lambda _out, _ctx, _i: 0,
        exit_condition_callable=_exit_after_four,
        max_loops=10,
    )
    limits = UsageLimits(total_cost_usd_limit=2.0)
    runner = Flujo(loop_step, usage_limits=limits)

    result = await gather_result(runner, 0)

    assert result.total_cost_usd == pytest.approx(0.8)
    assert len(result.step_history) == 1
    loop_result = result.step_history[0]
    assert loop_result.success
    assert loop_result.attempts == 4
    assert loop_result.cost_usd == pytest.approx(0.8)
