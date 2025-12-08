import asyncio
import time
from typing import Any

import pytest
from pydantic import BaseModel, Field

from flujo.application.core.hydration_manager import HydrationManager
from flujo.domain.interfaces import StateProvider
from flujo.domain.models import ContextReference


class _SlowProvider(StateProvider[Any]):
    def __init__(self, delay: float = 1.0) -> None:
        self.delay = delay
        self.load_calls: list[str] = []

    async def load(self, key: str) -> Any:
        self.load_calls.append(key)
        await asyncio.sleep(self.delay)
        return {"key": key}

    async def save(self, key: str, data: Any) -> None:
        return None


class _Ctx(BaseModel):
    a: ContextReference[dict[str, Any]] = Field(
        default_factory=lambda: ContextReference(provider_id="p", key="a")
    )
    b: ContextReference[dict[str, Any]] = Field(
        default_factory=lambda: ContextReference(provider_id="p", key="b")
    )
    c: ContextReference[dict[str, Any]] = Field(
        default_factory=lambda: ContextReference(provider_id="p", key="c")
    )
    d: ContextReference[dict[str, Any]] = Field(
        default_factory=lambda: ContextReference(provider_id="p", key="d")
    )
    e: ContextReference[dict[str, Any]] = Field(
        default_factory=lambda: ContextReference(provider_id="p", key="e")
    )


@pytest.mark.asyncio
async def test_hydration_manager_parallel_loading():
    """Hydration should load references in parallel, not serial."""

    provider = _SlowProvider(delay=1.0)
    hm = HydrationManager(state_providers={"p": provider})
    ctx = _Ctx()

    start = time.perf_counter()
    await hm.hydrate_context(ctx)
    duration = time.perf_counter() - start

    # Five loads at 1s each should complete ~1s if parallel (allow modest overhead)
    assert duration < 2.5, f"Hydration too slow; expected parallelism, got {duration:.2f}s"
    assert all(ref.get() is not None for ref in (ctx.a, ctx.b, ctx.c, ctx.d, ctx.e))
