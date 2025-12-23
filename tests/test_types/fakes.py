"""Typed fakes for Flujo tests.

These fakes provide lightweight, type-safe stand-ins for common runtime
collaborators (agents, usage meters, cache backends) without relying on
MagicMock/AsyncMock. They are intended to replace ad-hoc mocks in tests.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional

from flujo.application.core.executor_protocols import IUsageMeter, ICacheBackend
from flujo.domain.resources import AppResources
from flujo.domain.models import BaseModel, StepResult
from pydantic import Field


class FakeAgent:
    """Minimal async agent fake with call tracking."""

    def __init__(self, output: Any = "ok") -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

    async def run(
        self,
        data: Any,
        *,
        context: Any | None = None,
        resources: AppResources | None = None,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(
            {"data": data, "context": context, "resources": resources, "kwargs": kwargs}
        )
        return self.output


class TestContext(BaseModel):
    """Minimal typed context for tests."""

    counter: int = 0
    scratchpad: dict[str, object] = Field(default_factory=dict)
    step_outputs: dict[str, object] = Field(default_factory=dict)


class FakeUsageMeter(IUsageMeter):
    """Usage meter that records reservations without external side effects."""

    def __init__(self) -> None:
        self.reservations: list[dict[str, Any]] = []
        self.reconciliations: list[dict[str, Any]] = []
        self.snapshots: list[tuple[float, int, int]] = []

    async def add(self, cost_usd: float, prompt_tokens: int, completion_tokens: int) -> None:
        self.reservations.append(
            {
                "cost_usd": cost_usd,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )

    async def guard(self, limits: Any, step_history: list[Any] | None = None) -> None:
        self.reconciliations.append({"limits": limits, "step_history": step_history or []})

    async def snapshot(self) -> tuple[float, int, int]:
        current = (
            sum(item["cost_usd"] for item in self.reservations),
            sum(item["prompt_tokens"] for item in self.reservations),
            sum(item["completion_tokens"] for item in self.reservations),
        )
        self.snapshots.append(current)
        return current


__all__ = ["FakeAgent", "FakeUsageMeter", "FakeCacheBackend", "TestContext"]


class FakeCacheBackend(ICacheBackend):
    """In-memory cache backend for tests."""

    def __init__(self, *, max_size: int = 8) -> None:
        self.max_size = max_size
        self.store: OrderedDict[str, StepResult] = OrderedDict()

    async def get(self, key: str) -> Optional[StepResult]:
        value = self.store.get(key)
        if value is not None:
            try:
                self.store.move_to_end(key)
            except KeyError:
                pass
        return value

    async def put(self, key: str, value: StepResult, ttl_s: int) -> None:  # noqa: ARG002
        self.store[key] = value
        try:
            self.store.move_to_end(key)
        except KeyError:
            pass
        while len(self.store) > self.max_size:
            try:
                self.store.popitem(last=False)
            except KeyError:
                break

    async def clear(self) -> None:
        self.store.clear()
