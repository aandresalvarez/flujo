from __future__ import annotations

import pytest
from flujo.domain.models import PipelineContext


@pytest.mark.asyncio
async def test_import_step_outputs_none_merges_full_child_context() -> None:
    from flujo.domain.dsl.step import Step
    from flujo.domain.dsl.pipeline import Pipeline
    from flujo.application.runner import Flujo

    # PipelineContext imported at module scope for type hint resolution
    from flujo.testing.utils import gather_result
    from flujo.domain.dsl.import_step import ImportStep

    async def child_writer(_data: object, *, context: PipelineContext | None = None) -> dict:
        assert context is not None
        return {"scratchpad": {"foo": 42}}

    child = Pipeline.from_step(Step.from_callable(child_writer, name="child", updates_context=True))

    # outputs=None (default) means full child context is merged back
    import_step = ImportStep(
        name="run_child",
        pipeline=child,
        inherit_context=True,
        # outputs left as None
        updates_context=True,
    )
    parent = Pipeline.from_step(import_step)

    runner = Flujo(parent, context_model=PipelineContext)
    res = await gather_result(runner, "ignored")
    ctx = res.final_pipeline_context
    assert isinstance(ctx, PipelineContext)
    assert ctx.scratchpad.get("foo") == 42


@pytest.mark.asyncio
async def test_import_step_outputs_empty_list_merges_nothing() -> None:
    from flujo.domain.dsl.step import Step
    from flujo.domain.dsl.pipeline import Pipeline
    from flujo.application.runner import Flujo

    # PipelineContext imported at module scope for type hint resolution
    from flujo.testing.utils import gather_result
    from flujo.domain.dsl.import_step import ImportStep

    async def child_writer(_data: object, *, context: PipelineContext | None = None) -> dict:
        assert context is not None
        return {"scratchpad": {"bar": "no-merge"}}

    child = Pipeline.from_step(Step.from_callable(child_writer, name="child", updates_context=True))

    # Explicit empty outputs list → do not merge any fields back
    import_step = ImportStep(
        name="run_child",
        pipeline=child,
        inherit_context=True,
        outputs=[],
        updates_context=True,
    )
    parent = Pipeline.from_step(import_step)

    runner = Flujo(parent, context_model=PipelineContext)
    res = await gather_result(runner, "ignored")
    ctx = res.final_pipeline_context
    assert isinstance(ctx, PipelineContext)
    # Ensure nothing was merged from child
    assert "bar" not in ctx.scratchpad
