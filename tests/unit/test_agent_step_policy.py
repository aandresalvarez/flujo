import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.executor_helpers import make_execution_frame
from flujo.application.core.step_policies import DefaultAgentStepExecutor
from flujo.domain.dsl.step import Step
from flujo.domain.models import Success


@pytest.mark.asyncio
async def test_agent_quota_reservation_failure_raises_usage_limit_no_fallback():
    core = ExecutorCore()

    class _PrimaryAgent:
        async def run(self, *_args, **_kwargs):
            return "primary"

    class _FallbackAgent:
        async def run(self, *_args, **_kwargs):
            return "fallback"

    step = Step(name="primary", agent=_PrimaryAgent())
    # Attach a fallback to verify it is NOT used when reservation fails
    step.fallback_step = Step(name="fallback", agent=_FallbackAgent())

    # Inject a core estimator to over-estimate so reservation fails
    execu = DefaultAgentStepExecutor()
    from flujo.domain.models import UsageEstimate, Quota

    class _HighEstimator:
        def estimate(self, *_args):
            return UsageEstimate(cost_usd=100.0, tokens=0)

    core._usage_estimator = _HighEstimator()  # type: ignore[attr-defined]

    # Set a very small parent quota
    core._set_current_quota(Quota(remaining_cost_usd=1.0, remaining_tokens=1000))

    from flujo.exceptions import UsageLimitExceededError

    frame = make_execution_frame(
        core,
        step,
        data=None,
        context=None,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )

    with pytest.raises(UsageLimitExceededError) as exc:
        await execu.execute(core=core, frame=frame)
    # Message parity: generic when no explicit limits passed
    assert "Insufficient quota" in str(exc.value)


@pytest.mark.asyncio
async def test_agent_primary_failure_fallback_success_with_quota_present():
    core = ExecutorCore()

    class _PrimaryAgent:
        async def run(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    class _FallbackAgent:
        async def run(self, *_args, **_kwargs):
            return "ok"

    step = Step(name="primary", agent=_PrimaryAgent())
    step.fallback_step = Step(name="fallback", agent=_FallbackAgent())

    execu = DefaultAgentStepExecutor()
    # Inject low estimator so reservation succeeds
    from flujo.domain.models import UsageEstimate, Quota

    class _LowEstimator:
        def estimate(self, *_args):
            return UsageEstimate(cost_usd=0.0, tokens=0)

    core._usage_estimator = _LowEstimator()  # type: ignore[attr-defined]
    # Provide ample quota
    core._set_current_quota(Quota(remaining_cost_usd=10.0, remaining_tokens=1000))

    frame = make_execution_frame(
        core,
        step,
        data=None,
        context=None,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )

    outcome = await execu.execute(core=core, frame=frame)
    assert isinstance(outcome, Success)
    assert outcome.step_result.success is True
    assert outcome.step_result.metadata_.get("fallback_triggered") is True


@pytest.mark.asyncio
async def test_agent_quota_denial_uses_legacy_message_with_limits():
    core = ExecutorCore()

    class _PrimaryAgent:
        async def run(self, *_args, **_kwargs):
            return "primary"

    step = Step(name="primary", agent=_PrimaryAgent())
    execu = DefaultAgentStepExecutor()

    # Over-estimate so reservation fails; provide explicit limits for message parity
    from flujo.domain.models import UsageEstimate, Quota, UsageLimits

    class _HighEstimator:
        def estimate(self, *_args):
            return UsageEstimate(cost_usd=100.0, tokens=5_000_000)

    core._usage_estimator = _HighEstimator()  # type: ignore[attr-defined]
    core._set_current_quota(Quota(remaining_cost_usd=1.0, remaining_tokens=1))
    limits = UsageLimits(total_cost_usd_limit=1.0, total_tokens_limit=1)

    from flujo.exceptions import UsageLimitExceededError

    frame = make_execution_frame(
        core,
        step,
        data=None,
        context=None,
        resources=None,
        limits=limits,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )

    with pytest.raises(UsageLimitExceededError) as exc:
        await execu.execute(core=core, frame=frame)
    # When both exceeded, cost prioritization or most constrained ratio should pick cost here
    assert (
        str(exc.value) == "Cost limit of $1 exceeded"
        or str(exc.value) == "Cost limit of $1.0 exceeded"
    )


@pytest.mark.asyncio
async def test_agent_pricing_not_configured_raises_immediately():
    """Strict pricing errors must propagate without fallback or wrapping."""
    core = ExecutorCore()

    class _PrimaryAgent:
        async def run(self, *_args, **_kwargs):
            from flujo.exceptions import PricingNotConfiguredError

            raise PricingNotConfiguredError(provider="prov", model="m")

    step = Step(name="primary", agent=_PrimaryAgent())
    execu = DefaultAgentStepExecutor()

    from flujo.exceptions import PricingNotConfiguredError

    frame = make_execution_frame(
        core,
        step,
        data=None,
        context=None,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )

    with pytest.raises(PricingNotConfiguredError):
        await execu.execute(core=core, frame=frame)
