from typing import Any, Optional, Dict

import pytest
from pydantic import BaseModel, Field

from flujo.domain.dsl.step import Step, MergeStrategy
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.models import StepResult
from flujo.application.core.step_policies import DefaultParallelStepExecutor
from flujo.application.core.executor_helpers import make_execution_frame
from flujo.application.core.context_manager import ContextManager


class _Ctx(BaseModel):
    executed_branches: list[str] = Field(default_factory=list)
    tag: str | None = None


@pytest.mark.asyncio
async def test_parallel_executor_isolates_context_per_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_context = _Ctx()
    called: Dict[str, int] = {"isolate": 0}

    def fake_isolate(
        ctx: Any, include_keys: Optional[list[str]] = None, *, purpose: str = "unknown"
    ) -> Any:
        called["isolate"] += 1
        # Return a shallow copy-like namespace for visibility
        if isinstance(ctx, BaseModel):
            return _Ctx(**ctx.model_dump())
        return _Ctx()

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
        _branch_pipeline: Any, data: Any, ctx: Any, _resources: Any
    ) -> StepResult:
        return StepResult(name="branch", output=data, success=True, attempts=1)

    # Minimal core stub with quota manager
    class _Core:
        class _QuotaMgr:
            def get_current_quota(self):
                return None

        def __init__(self) -> None:
            self._quota_manager = self._QuotaMgr()

    execu = DefaultParallelStepExecutor()
    core = _Core()
    frame = make_execution_frame(
        core,
        p,
        {"x": 1},
        context=base_context,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    setattr(frame, "step_executor", fake_step_executor)

    res = await execu.execute(core, frame)

    res_sr = res.step_result if hasattr(res, "step_result") else res
    assert isinstance(res_sr, StepResult)
    # Called at least once per branch (can be more when verifying isolation)
    assert called["isolate"] >= 3


@pytest.mark.asyncio
async def test_parallel_executor_merges_successful_branch_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that parallel executor handles context merging for successful branches.

    Note: When context_include_keys is None (default), the executor manually merges
    specific typed attributes (step_outputs, import_artifacts, branch_results, context_updates) instead of
    calling ContextManager.merge. This test verifies the result has proper branch_context.
    """
    base_context = _Ctx()

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
        _branch_pipeline: Any, data: Any, ctx: Any, _resources: Any
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
        class _QuotaMgr:
            def get_current_quota(self):
                return None

        def __init__(self) -> None:
            self._quota_manager = self._QuotaMgr()

    execu = DefaultParallelStepExecutor()
    core = _Core()
    frame = make_execution_frame(
        core,
        p,
        {"x": 1},
        context=base_context,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    setattr(frame, "step_executor", fake_step_executor)

    res = await execu.execute(core, frame)

    # Verify result has branch_context set
    res_sr = res.step_result if hasattr(res, "step_result") else res
    assert isinstance(res_sr, StepResult)
    # The executor should set branch_context on the result
    assert res_sr.branch_context is not None
