import pytest

from typing import Any

from flujo import Step, Flujo
from flujo.caching import InMemoryCache
from flujo.steps.cache_step import _generate_cache_key
from flujo.testing.utils import StubAgent, gather_result
from pydantic import BaseModel


class Model(BaseModel):
    a: int
    b: str


def test_generate_cache_key() -> None:
    m = Model(a=1, b="x")
    key1 = _generate_cache_key("step", m)
    key2 = _generate_cache_key("step", {"a": 1, "b": "x"})
    assert isinstance(key1, str)
    assert isinstance(key2, str)

    class Unserializable:
        pass

    assert _generate_cache_key("step", Unserializable()) is None

    key_ctx1 = _generate_cache_key("step", m, context={"val": 1})
    key_ctx2 = _generate_cache_key("step", m, context={"val": 2})
    assert key_ctx1 != key_ctx2


@pytest.mark.asyncio
async def test_cache_hit_and_miss() -> None:
    agent = StubAgent(["ok"])
    inner = Step.solution(agent)
    cache = InMemoryCache()
    cached_step = Step.cached(inner, cache_backend=cache)
    runner = Flujo(cached_step)

    result1 = await gather_result(runner, "in")
    first_meta = result1.step_history[0].metadata_
    result2 = await gather_result(runner, "in")

    assert agent.call_count == 1
    assert first_meta is None or "cache_hit" not in first_meta
    assert result2.step_history[0].metadata_["cache_hit"] is True


class FailingBackend(InMemoryCache):
    async def get(self, key: str) -> Any:
        raise RuntimeError("boom")

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_cache_backend_errors_are_ignored() -> None:
    agent = StubAgent(["ok", "ok"])
    inner = Step.solution(agent)
    cached = Step.cached(inner, cache_backend=FailingBackend())
    runner = Flujo(cached)

    await gather_result(runner, "x")
    await gather_result(runner, "x")

    assert agent.call_count == 2


@pytest.mark.asyncio
async def test_cache_key_differs_for_same_name_steps() -> None:
    agent1 = StubAgent(["a", "a"])
    agent2 = StubAgent(["b", "b"])
    step1 = Step.solution(agent1, name="dup")
    step2 = Step.solution(agent2, name="dup")
    cache = InMemoryCache()
    runner = Flujo(
        Step.cached(step1, cache_backend=cache) >> Step.cached(step2, cache_backend=cache)
    )

    await gather_result(runner, "in")
    result2 = await gather_result(runner, "in")

    h1, h2 = result2.step_history
    assert h1.metadata_["cache_hit"] is True
    assert h2.metadata_["cache_hit"] is True
    assert agent1.call_count == 1
    assert agent2.call_count == 1
