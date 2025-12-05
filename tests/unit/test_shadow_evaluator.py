from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from flujo.application.core.shadow_evaluator import ShadowEvalConfig, ShadowEvaluator


class DummyBg:
    def __init__(self) -> None:
        self.added: list[Any] = []

    def add_task(self, task: Any) -> None:
        self.added.append(task)


@pytest.mark.asyncio
async def test_shadow_eval_schedules_when_enabled(monkeypatch: Any) -> None:
    created: list[Any] = []

    async def fake_create_task(coro: Any, name: str | None = None) -> Any:
        created.append(coro)

        class DummyTask:
            def add_done_callback(self, _: Any) -> None:
                pass

        return DummyTask()

    monkeypatch.setattr(asyncio, "create_task", fake_create_task)
    monkeypatch.setattr(
        "flujo.application.core.shadow_evaluator.asyncio.create_task", fake_create_task
    )
    monkeypatch.setattr("flujo.application.core.shadow_evaluator.random.random", lambda: 0.0)

    evaluator = ShadowEvaluator(
        config=ShadowEvalConfig(
            enabled=True,
            sample_rate=1.0,
            timeout_s=0.1,
            judge_model="test-model",
            sink="telemetry",
        ),
        background_task_manager=DummyBg(),
    )

    evaluator._run_judge = lambda **_: asyncio.sleep(0)  # type: ignore[assignment]
    evaluator.maybe_schedule(
        core=object(), step=SimpleNamespace(name="s1"), result=SimpleNamespace(success=True)
    )

    assert evaluator._sampled == 1  # type: ignore[attr-defined]


def test_shadow_eval_disabled(monkeypatch: Any) -> None:
    created: list[Any] = []

    async def fake_create_task(coro: Any, name: str | None = None) -> Any:
        created.append(coro)

    monkeypatch.setattr(asyncio, "create_task", fake_create_task)

    evaluator = ShadowEvaluator(
        config=ShadowEvalConfig(
            enabled=False,
            sample_rate=0.0,
            timeout_s=0.1,
            judge_model="test-model",
            sink="telemetry",
        ),
        background_task_manager=DummyBg(),
    )

    evaluator.maybe_schedule(
        core=object(), step=SimpleNamespace(name="s1"), result=SimpleNamespace(success=True)
    )
    assert created == []
