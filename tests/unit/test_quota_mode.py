import pytest

from flujo.domain.dsl.step import Step
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.models import UsageLimits
from flujo.application.runner import Flujo
from flujo.exceptions import UsageLimitExceededError


class _Cfg:
    def __init__(self, expected_cost_usd: float = 0.0, expected_tokens: int = 0) -> None:
        self.expected_cost_usd = expected_cost_usd
        self.expected_tokens = expected_tokens


async def _noop_agent(
    payload: str,
    *,
    context=None,
    resources=None,
    options=None,
    stream=False,
    on_chunk=None,
):
    class _R:
        def usage(self):
            class U:
                def __init__(self):
                    self.request_tokens = 0
                    self.response_tokens = 1

            return U()

    return _R()


@pytest.mark.asyncio
async def test_pure_quota_reservation_denies(monkeypatch):
    monkeypatch.setenv("FLUJO_PURE_QUOTA", "1")
    step = Step(name="s", agent=type("A", (), {"run": staticmethod(_noop_agent)})())
    step.config = _Cfg(expected_cost_usd=1.0, expected_tokens=100)
    runner = Flujo(
        pipeline=step, usage_limits=UsageLimits(total_cost_usd_limit=0.5, total_tokens_limit=10)
    )
    with pytest.raises(UsageLimitExceededError):
        # Consume the single outcome from run_outcomes_async
        async for _ in runner.run_outcomes_async("x"):
            pass


@pytest.mark.asyncio
async def test_pure_quota_parallel_split_insufficient(monkeypatch):
    monkeypatch.setenv("FLUJO_PURE_QUOTA", "1")

    def mkstep(n):
        s = Step(name=f"s{n}", agent=type("A", (), {"run": staticmethod(_noop_agent)})())
        s.config = _Cfg(expected_cost_usd=1.0, expected_tokens=100)
        return s

    p = ParallelStep(name="p", branches={"a": mkstep(1), "b": mkstep(2)})
    runner = Flujo(
        pipeline=p, usage_limits=UsageLimits(total_cost_usd_limit=1.0, total_tokens_limit=50)
    )
    with pytest.raises(UsageLimitExceededError):
        async for _ in runner.run_outcomes_async("x"):
            pass
