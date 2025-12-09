"""Typed fakes for Flujo tests.

These fakes provide lightweight, type-safe stand-ins for common runtime
collaborators (agents, usage meters, cache backends) without relying on
MagicMock/AsyncMock. They are intended to replace ad-hoc mocks in tests.
"""

from __future__ import annotations

from typing import Any

from flujo.application.core.executor_protocols import IUsageMeter
from flujo.domain.resources import AppResources


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


__all__ = ["FakeAgent", "FakeUsageMeter"]
