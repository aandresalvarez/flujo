import asyncio

import pytest

from flujo.application.core.policies.cache_policy import DefaultCacheStepExecutor
from flujo.domain.dsl.cache_step import CacheStep
from flujo.domain.dsl import Step
from flujo.domain.models import StepResult, Success
from flujo.application.core.types import ExecutionFrame


class _FakeCacheBackend:
    def __init__(self) -> None:
        self.get_calls = 0

    async def get(self, key: str):
        self.get_calls += 1
        return StepResult(
            name="cached_step",
            success=True,
            output="cached",
            metadata_={},
            branch_context=None,
        )

    async def set(self, key: str, value):
        return None


class _FakeCore:
    def __init__(self) -> None:
        self._cache_locks_lock = asyncio.Lock()
        self._cache_locks: dict[str, asyncio.Lock] = {}

    def _get_cache_locks_lock(self) -> asyncio.Lock:
        return self._cache_locks_lock

    def _get_current_quota(self):
        return None

    async def execute(self, frame):
        return Success(result=StepResult(name="cached_step", success=True, output="miss"))


@pytest.mark.asyncio
async def test_loop_step_cache_disablement_scope():
    """CacheStep should still hit cache when sharing core with loop branches."""

    backend = _FakeCacheBackend()
    wrapped = Step.from_callable(lambda x: "miss", name="cached_step")
    cache_step = CacheStep.cached(wrapped_step=wrapped, cache_backend=backend)
    core = _FakeCore()

    frame = ExecutionFrame(
        step=cache_step,
        data="payload",
        context=None,
        resources=None,
        limits=None,
        quota=None,
        stream=False,
        on_chunk=None,
        context_setter=lambda _pr, _ctx: None,
        result=None,
    )

    outcome = await DefaultCacheStepExecutor().execute(core=core, frame=frame)
    assert isinstance(outcome, Success)
    assert outcome.step_result.output == "cached"
    assert outcome.step_result.metadata_.get("cache_hit") is True
    assert backend.get_calls == 1
