import pytest
from unittest.mock import MagicMock
from flujo import (
    Flujo,
    Step,
    Pipeline,
    AppResources,
    UsageLimits,
)
from flujo.exceptions import PipelineAbortSignal, UsageLimitExceededError
from flujo.domain.models import PipelineResult
from pydantic import BaseModel
from typing import Any, List, cast
from flujo.domain.events import (
    HookPayload,
    PreRunPayload,
    PostStepPayload,
)
from flujo.domain.agent_protocol import AsyncAgentProtocol
from flujo.testing.utils import StubAgent, DummyPlugin, gather_result
from flujo.domain.plugins import PluginOutcome
from tests.integration.test_usage_governor import FixedMetricAgent


class HookResources(AppResources):
    db: MagicMock


class HookContext(BaseModel):
    call_count: int = 0


@pytest.fixture  # type: ignore[misc]
def call_recorder() -> List[HookPayload]:
    return []


async def generic_recorder_hook(call_recorder: List[HookPayload], payload: HookPayload) -> None:
    call_recorder.append(payload)
    context = getattr(payload, "pipeline_context", None)
    if context is not None:
        context.call_count += 1


async def aborting_hook(call_recorder: List[HookPayload], payload: HookPayload) -> None:
    if payload.event_name == "on_step_failure":
        call_recorder.append(payload)
        raise PipelineAbortSignal("Aborted from hook")


async def erroring_hook(payload: HookPayload) -> None:
    raise ValueError("Hook failed!")


async def post_run_abort_hook(payload: HookPayload) -> None:
    if payload.event_name == "post_run":
        raise PipelineAbortSignal("abort in post_run")


@pytest.mark.asyncio  # type: ignore[misc]
async def test_all_hooks_are_called_in_correct_order(
    call_recorder: List[HookPayload],
) -> None:
    pipeline = Step(
        "s1",
        agent=cast(AsyncAgentProtocol[Any, Any], StubAgent(["ok1"])),
    ) >> Step("s2", agent=cast(AsyncAgentProtocol[Any, Any], StubAgent(["ok2"])))

    async def recorder(payload: HookPayload) -> None:
        await generic_recorder_hook(call_recorder, payload)

    runner = Flujo(pipeline, hooks=[recorder])
    await gather_result(runner, "start")

    events = [p.event_name for p in call_recorder]
    assert events == [
        "pre_run",
        "pre_step",
        "post_step",
        "pre_step",
        "post_step",
        "post_run",
    ]


@pytest.mark.asyncio  # type: ignore[misc]
async def test_on_step_failure_hook_is_called(
    call_recorder: List[HookPayload],
) -> None:
    failing_plugin = DummyPlugin([PluginOutcome(success=False)])
    pipeline = Step("s1", agent=cast(AsyncAgentProtocol[Any, Any], StubAgent(["ok"]))) >> Step(
        "s2",
        agent=cast(AsyncAgentProtocol[Any, Any], StubAgent(["bad"])),
        plugins=[failing_plugin],
    )

    async def recorder(payload: HookPayload) -> None:
        await generic_recorder_hook(call_recorder, payload)

    runner = Flujo(pipeline, hooks=[recorder])
    await gather_result(runner, "start")

    events = [p.event_name for p in call_recorder]
    assert events == [
        "pre_run",
        "pre_step",
        "post_step",
        "pre_step",
        "on_step_failure",
        "post_run",
    ]


@pytest.mark.asyncio  # type: ignore[misc]
async def test_hook_receives_correct_arguments(
    call_recorder: List[HookPayload],
) -> None:
    pipeline = Step("s1", agent=cast(AsyncAgentProtocol[Any, Any], StubAgent(["ok"])))

    async def recorder(payload: HookPayload) -> None:
        await generic_recorder_hook(call_recorder, payload)

    runner = Flujo(pipeline, hooks=[recorder])
    await gather_result(runner, "start")

    pre_run_call = next(p for p in call_recorder if p.event_name == "pre_run")
    assert isinstance(pre_run_call, PreRunPayload)

    post_step_call = next(p for p in call_recorder if p.event_name == "post_step")
    assert isinstance(post_step_call, PostStepPayload)


@pytest.mark.asyncio  # type: ignore[misc]
async def test_pipeline_aborts_gracefully_from_hook(
    call_recorder: List[HookPayload],
) -> None:
    failing_plugin = DummyPlugin([PluginOutcome(success=False)])
    pipeline = (
        Step("s1", agent=cast(AsyncAgentProtocol[Any, Any], StubAgent(["ok"])))
        >> Step(
            "s2",
            agent=cast(AsyncAgentProtocol[Any, Any], StubAgent(["bad"])),
            plugins=[failing_plugin],
        )
        >> Step("s3", agent=cast(AsyncAgentProtocol[Any, Any], StubAgent(["unused"])))
    )

    async def hook(payload: HookPayload) -> None:
        await aborting_hook(call_recorder, payload)

    runner = Flujo(pipeline, hooks=[hook])
    result = await gather_result(runner, "start")

    assert isinstance(result, PipelineResult)
    assert len(result.step_history) == 2
    assert result.step_history[0].success is True
    assert result.step_history[1].success is False


@pytest.mark.asyncio  # type: ignore[misc]
async def test_faulty_hook_does_not_crash_pipeline(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline = Step("s1", agent=cast(AsyncAgentProtocol[Any, Any], StubAgent(["ok"])))
    runner = Flujo(pipeline, hooks=[erroring_hook])

    result = await gather_result(runner, "start")

    assert result.step_history[0].success is True
    assert "Error in hook" in caplog.text
    assert "Hook failed!" in caplog.text


@pytest.mark.asyncio  # type: ignore[misc]
async def test_hooks_receive_context_and_resources(
    call_recorder: List[HookPayload],
) -> None:
    pipeline = Step("s1", agent=cast(AsyncAgentProtocol[Any, Any], StubAgent(["ok"])))
    mock_res = HookResources(db=MagicMock())

    async def recorder(payload: HookPayload) -> None:
        await generic_recorder_hook(call_recorder, payload)

    runner = Flujo(
        pipeline,
        context_model=HookContext,
        initial_context_data={"call_count": 0},
        resources=mock_res,
        hooks=[recorder],
    )
    result = await gather_result(runner, "start")

    post_step_call = next(p for p in call_recorder if p.event_name == "post_step")
    assert getattr(post_step_call, "pipeline_context") is not None
    assert getattr(post_step_call, "resources") is not None
    assert isinstance(result.final_pipeline_context, HookContext)
    assert result.final_pipeline_context.call_count > 0


@pytest.mark.asyncio  # type: ignore[misc]
async def test_post_run_abort_does_not_mask_errors() -> None:
    """Abort signal in post_run should not hide UsageLimitExceededError."""
    limits = UsageLimits(total_cost_usd_limit=0.0, total_tokens_limit=None)
    pipeline = Pipeline.from_step(Step("metric_step", FixedMetricAgent()))
    runner = Flujo(pipeline, usage_limits=limits, hooks=[post_run_abort_hook])

    with pytest.raises(UsageLimitExceededError):
        await gather_result(runner, 0)
