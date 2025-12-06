import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.executor_helpers import make_execution_frame
from flujo.application.core.step_policies import DefaultAgentStepExecutor
from flujo.domain.dsl.step import Step


@pytest.mark.asyncio
async def test_agent_strict_pricing_error_surfaces_post_reservation():
    core = ExecutorCore()

    class _Agent:
        async def run(self, *_args, **_kwargs):
            # Return an object to trigger usage extraction
            class _Out:
                pass

            return _Out()

    step = Step(name="primary", agent=_Agent())

    execu = DefaultAgentStepExecutor()
    # Ensure reservation succeeds (low estimate)
    from flujo.domain.models import UsageEstimate, Quota

    def _fake_estimate(_s, _d, _c):
        return UsageEstimate(cost_usd=0.0, tokens=0)

    execu._estimate_usage = _fake_estimate  # type: ignore[attr-defined]
    core._set_current_quota(Quota(remaining_cost_usd=1.0, remaining_tokens=1000))

    # Patch extract_usage_metrics to raise strict pricing error
    import flujo.application.core.step_policies as policies_mod

    real_extract = policies_mod.extract_usage_metrics

    def _raise_pricing(*_a, **_k):
        # Simulate strict pricing error structure used by the system
        class _FakeErr(Exception):
            pass

        raise _FakeErr("strict pricing not configured for test")

    policies_mod.extract_usage_metrics = _raise_pricing
    try:
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
        from flujo.domain.models import Failure as OutcomeFailure

        assert isinstance(outcome, OutcomeFailure)
        # Feedback should include strict pricing message
        fb = (outcome.feedback or "") if hasattr(outcome, "feedback") else ""
        sr_fb = outcome.step_result.feedback if outcome.step_result else ""
        assert ("strict pricing" in fb.lower()) or (sr_fb and "strict pricing" in sr_fb.lower())
    finally:
        policies_mod.extract_usage_metrics = real_extract
