from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from flujo.application.core.shadow_evaluator import ShadowEvalConfig, ShadowEvaluator
from flujo.domain.evaluation import EvaluationScore


class DummyBg:
    def __init__(self) -> None:
        self.added: list[Any] = []

    def add_task(self, task: Any) -> None:
        self.added.append(task)


@pytest.mark.asyncio
async def test_shadow_eval_schedules_when_enabled(monkeypatch: Any) -> None:
    created: list[Any] = []

    def fake_create_task(coro: Any, name: str | None = None) -> Any:
        created.append(coro)
        try:
            coro.close()
        except Exception:
            pass

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

    def fake_create_task(coro: Any, name: str | None = None) -> Any:
        created.append(coro)
        try:
            coro.close()
        except Exception:
            pass

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


@pytest.mark.asyncio
async def test_run_judge_invokes_agent(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    class DummyAgent:
        def __init__(self) -> None:
            self.seen = calls

        async def run(self, payload: dict[str, Any]) -> EvaluationScore:
            self.seen.append(payload)
            return EvaluationScore(score=0.8, reasoning="ok", criteria={"quality": 0.8})

    def fake_make_agent_async(**_: Any) -> DummyAgent:
        return DummyAgent()

    monkeypatch.setattr(
        "flujo.application.core.shadow_evaluator.make_agent_async", fake_make_agent_async
    )

    evaluator = ShadowEvaluator(
        config=ShadowEvalConfig(
            enabled=True,
            sample_rate=1.0,
            timeout_s=1.0,
            judge_model="test-model",
            sink="telemetry",
        ),
        background_task_manager=DummyBg(),
    )

    await evaluator._run_judge(core=object(), payload={"step_name": "s1", "output": "x"})

    assert calls and calls[0]["step_name"] == "s1"
