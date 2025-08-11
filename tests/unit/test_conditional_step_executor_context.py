from typing import Any

import pytest

from flujo.application.core.context_manager import ContextManager
from flujo.application.core.step_policies import DefaultConditionalStepExecutor
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step


@pytest.mark.asyncio
async def test_conditional_executor_isolates_and_merges(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"isolate": 0, "merge": 0}

    def fake_isolate(ctx: Any) -> Any:
        calls["isolate"] += 1
        return ctx

    def fake_merge(main_ctx: Any, branch_ctx: Any) -> Any:
        calls["merge"] += 1
        return main_ctx

    monkeypatch.setattr(ContextManager, "isolate", staticmethod(fake_isolate))
    monkeypatch.setattr(ContextManager, "merge", staticmethod(fake_merge))

    # Build a simple conditional with two branches
    async def a(x: Any) -> Any:
        return x

    async def b(x: Any) -> Any:
        return x

    cond = ConditionalStep(
        name="c",
        condition_callable=lambda _d, _c: "A",
        branches={
            "A": Pipeline.from_step(Step.from_callable(a, name="A")),
            "B": Pipeline.from_step(Step.from_callable(b, name="B")),
        },
    )

    class _Core:
        async def execute(
            self,
            step,
            data,
            context=None,
            resources=None,
            limits=None,
            context_setter=None,
            _fallback_depth=0,
        ):  # type: ignore[no-redef]
            # Return a StepResult-like object for the selected branch
            from flujo.domain.models import StepResult

            return StepResult(name="A", output=data, success=True, branch_context=context)

    execu = DefaultConditionalStepExecutor()
    res = await execu.execute(
        _Core(),
        cond,
        data=1,
        context=type("C", (), {})(),
        resources=None,
        limits=None,
        context_setter=None,
    )

    assert res.success is True
    assert calls["isolate"] == 1
    assert calls["merge"] == 1
