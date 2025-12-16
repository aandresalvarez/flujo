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
            evaluate_on_failure=False,
        ),
        background_task_manager=DummyBg(),
    )

    evaluator._run_judge = lambda **_: asyncio.sleep(0)  # type: ignore[method-assign]
    evaluator.maybe_schedule(
        core=object(), step=SimpleNamespace(name="s1"), result=SimpleNamespace(success=True)
    )

    assert evaluator._sampled == 1


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
            evaluate_on_failure=False,
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
            evaluate_on_failure=False,
        ),
        background_task_manager=DummyBg(),
    )

    await evaluator._run_judge(core=object(), payload={"step_name": "s1", "output": "x"})

    assert calls and calls[0]["step_name"] == "s1"


@pytest.mark.asyncio
async def test_shadow_eval_evaluate_on_failure_only(monkeypatch: Any) -> None:
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
            evaluate_on_failure=True,
        ),
        background_task_manager=DummyBg(),
    )

    evaluator._run_judge = lambda **_: asyncio.sleep(0)  # type: ignore[method-assign]

    evaluator.maybe_schedule(
        core=object(),
        step=SimpleNamespace(name="s1"),
        result=SimpleNamespace(success=True),
        frame=SimpleNamespace(context=SimpleNamespace(run_id="run_x")),
    )
    evaluator.maybe_schedule(
        core=object(),
        step=SimpleNamespace(name="s2"),
        result=SimpleNamespace(success=False),
        frame=SimpleNamespace(context=SimpleNamespace(run_id="run_x")),
    )
    assert len(created) == 1


def test_shadow_eval_sampling_is_cached_per_run(monkeypatch: Any) -> None:
    seq = iter([0.9, 0.1])
    monkeypatch.setattr("flujo.application.core.shadow_evaluator.random.random", lambda: next(seq))

    evaluator = ShadowEvaluator(
        config=ShadowEvalConfig(
            enabled=True,
            sample_rate=0.5,
            timeout_s=0.1,
            judge_model="test-model",
            sink="telemetry",
            evaluate_on_failure=False,
        ),
        background_task_manager=DummyBg(),
    )

    evaluator.maybe_schedule(
        core=object(),
        step=SimpleNamespace(name="s1"),
        result=SimpleNamespace(success=False),
        frame=SimpleNamespace(context=SimpleNamespace(run_id="run_x")),
    )
    evaluator.maybe_schedule(
        core=object(),
        step=SimpleNamespace(name="s2"),
        result=SimpleNamespace(success=False),
        frame=SimpleNamespace(context=SimpleNamespace(run_id="run_x")),
    )

    assert next(seq) == 0.1


@pytest.mark.asyncio
async def test_shadow_eval_run_level_schedules_once(monkeypatch: Any) -> None:
    monkeypatch.setattr("flujo.application.core.shadow_evaluator.random.random", lambda: 0.0)

    bg = DummyBg()
    evaluator = ShadowEvaluator(
        config=ShadowEvalConfig(
            enabled=True,
            sample_rate=1.0,
            timeout_s=1.0,
            judge_model="test-model",
            sink="telemetry",
            evaluate_on_failure=False,
            run_level_enabled=True,
        ),
        background_task_manager=bg,
    )

    payloads: list[dict[str, Any]] = []

    async def fake_run_judge(*, core: object, payload: dict[str, object]) -> None:
        payloads.append(dict(payload))
        await asyncio.sleep(0)

    evaluator._run_judge = fake_run_judge  # type: ignore[method-assign]

    core = SimpleNamespace(context=SimpleNamespace(run_id="run_x"))
    result = SimpleNamespace(
        success=True,
        status="completed",
        output="ok",
        total_cost_usd=0.0,
        step_history=[SimpleNamespace(name="s1", success=True, feedback=None)],
        final_pipeline_context=SimpleNamespace(run_id="run_x"),
    )

    evaluator.maybe_schedule_run(core=core, result=result, run_id="run_x")
    evaluator.maybe_schedule_run(core=core, result=result, run_id="run_x")

    if bg.added:
        await asyncio.gather(*bg.added)

    assert evaluator._sampled == 1
    assert len(payloads) == 1
    assert payloads[0]["step_name"] == "__run__"


def test_shadow_eval_run_level_respects_failure_only(monkeypatch: Any) -> None:
    monkeypatch.setattr("flujo.application.core.shadow_evaluator.random.random", lambda: 0.0)

    evaluator = ShadowEvaluator(
        config=ShadowEvalConfig(
            enabled=True,
            sample_rate=1.0,
            timeout_s=0.1,
            judge_model="test-model",
            sink="telemetry",
            evaluate_on_failure=True,
            run_level_enabled=True,
        ),
        background_task_manager=DummyBg(),
    )

    evaluator.maybe_schedule_run(
        core=SimpleNamespace(context=SimpleNamespace(run_id="run_x")),
        result=SimpleNamespace(success=True, step_history=[]),
        run_id="run_x",
    )
    assert evaluator._sampled == 0
