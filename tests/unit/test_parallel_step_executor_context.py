import types
from typing import Any, Optional, Dict

import pytest

from flujo.domain.dsl.step import Step, MergeStrategy
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.models import StepResult
from flujo.application.core.step_policies import DefaultParallelStepExecutor
from flujo.application.core.context_manager import ContextManager


class _Ctx(types.SimpleNamespace):
    pass


@pytest.mark.asyncio
async def test_parallel_executor_isolates_context_per_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_context = _Ctx()
    called: Dict[str, int] = {"isolate": 0}

    def fake_isolate(ctx: Any, include_keys: Optional[list[str]] = None) -> Any:
        called["isolate"] += 1
        # Return a shallow copy-like namespace for visibility
        return _Ctx(**getattr(ctx, "__dict__", {}))

    monkeypatch.setattr(ContextManager, "isolate", staticmethod(fake_isolate))

    # Build a simple parallel with 3 branches
    async def _noop(x: Any) -> Any:
        return x

    p = ParallelStep(
        name="p",
        branches={
            "a": Pipeline.from_step(Step.from_callable(_noop, name="a")),
            "b": Pipeline.from_step(Step.from_callable(_noop, name="b")),
            "c": Pipeline.from_step(Step.from_callable(_noop, name="c")),
        },
        merge_strategy=MergeStrategy.NO_MERGE,
    )

    # Provide a fake step executor to avoid invoking core._execute_pipeline
    async def fake_step_executor(
        _branch_pipeline: Any, data: Any, ctx: Any, _resources: Any, _breach: Any
    ) -> StepResult:
        return StepResult(name="branch", output=data, success=True, attempts=1)

    # Minimal core stub
    class _Core:
        _ParallelUsageGovernor = lambda self, _limits: None  # type: ignore

    execu = DefaultParallelStepExecutor()
    res = await execu.execute(
        _Core(),
        p,
        data={"x": 1},
        context=base_context,
        resources=None,
        limits=None,
        breach_event=None,
        context_setter=None,
        step_executor=fake_step_executor,
    )

    assert isinstance(res, StepResult)
    # Called once per branch
    assert called["isolate"] == 3


@pytest.mark.asyncio
async def test_parallel_executor_merges_successful_branch_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_context = _Ctx()
    calls: Dict[str, int] = {"merge": 0}

    def fake_merge(main_ctx: Any, branch_ctx: Any) -> Any:
        calls["merge"] += 1
        return main_ctx

    monkeypatch.setattr(ContextManager, "merge", staticmethod(fake_merge))

    # Build a parallel with 2 successful branches and 1 failed branch
    async def _ok(x: Any) -> Any:
        return x

    async def _fail(x: Any) -> Any:
        raise RuntimeError("boom")

    p = ParallelStep(
        name="p",
        branches={
            "a": Pipeline.from_step(Step.from_callable(_ok, name="a")),
            "b": Pipeline.from_step(Step.from_callable(_ok, name="b")),
            "c": Pipeline.from_step(Step.from_callable(_fail, name="c")),
        },
        merge_strategy=MergeStrategy.CONTEXT_UPDATE,
    )

    # Provide a fake step executor that yields StepResult with branch_context for ok branches
    async def fake_step_executor(
        _branch_pipeline: Any, data: Any, ctx: Any, _resources: Any, _breach: Any
    ) -> StepResult:
        # Determine branch by probing pipeline name on first step, fallback to generic
        try:
            step0 = _branch_pipeline.steps[0]
            nm = getattr(step0, "name", "")
        except Exception:
            nm = ""
        if nm == "c":
            return StepResult(name="c", output=None, success=False)
        # Successful branch with a distinct branch context
        branch_ctx = _Ctx(tag=nm)
        return StepResult(name=nm, output=data, success=True, branch_context=branch_ctx)

    class _Core:
        _ParallelUsageGovernor = lambda self, _limits: None  # type: ignore

    execu = DefaultParallelStepExecutor()
    await execu.execute(
        _Core(),
        p,
        data={"x": 1},
        context=base_context,
        resources=None,
        limits=None,
        breach_event=None,
        context_setter=None,
        step_executor=fake_step_executor,
    )

    # merge should be called for each successful branch (a, b) only
    assert calls["merge"] == 2
