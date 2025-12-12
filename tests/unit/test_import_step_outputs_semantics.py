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
        return {"import_artifacts": {"foo": 42}}

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
    assert ctx.import_artifacts.get("foo") == 42


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
        return {"import_artifacts": {"bar": "no-merge"}}

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
    assert ctx.import_artifacts.get("bar") is None


@pytest.mark.asyncio
async def test_import_step_outputs_preserves_explicit_none_values() -> None:
    """Test that explicit None values in child context are preserved, not replaced.

    Regression test for: _get_child cannot distinguish between "path not found"
    and "path found with None value". When child context has an explicit None,
    it should NOT fall back to checking inner_sr.output.

    Scenario:
    - Child pipeline has two steps
    - First step sets import_artifacts.value to None (explicit null)
    - Second step's OUTPUT (not context) has import_artifacts.value = "from_output"
    - When mapping outputs, the None from context should be used, NOT the output
    """
    from flujo.domain.dsl.step import Step
    from flujo.domain.dsl.pipeline import Pipeline
    from flujo.application.runner import Flujo
    from flujo.testing.utils import gather_result
    from flujo.domain.dsl.import_step import ImportStep, OutputMapping

    async def step1_sets_none(_data: object, *, context: PipelineContext | None = None) -> dict:
        """First step explicitly sets value to None in context."""
        assert context is not None
        return {"import_artifacts": {"value": None, "marker": "step1_ran"}}

    async def step2_returns_different(
        _data: object, *, context: PipelineContext | None = None
    ) -> dict:
        """Second step returns a dict with 'value' in its output (but doesn't update context).

        This tests that _get_child prefers the context (which has None) over the
        last step's output (which has "from_output").
        """
        # This step does NOT have updates_context=True, so this output won't merge
        # But inner_sr.output will contain this value
        return {"import_artifacts": {"value": "from_output", "step2_marker": "step2_ran"}}

    child = Pipeline(
        steps=[
            Step.from_callable(step1_sets_none, name="step1", updates_context=True),
            Step.from_callable(step2_returns_different, name="step2", updates_context=False),
        ]
    )

    # Create import step that maps child's import_artifacts.value to parent import artifacts
    import_step = ImportStep(
        name="run_child",
        pipeline=child,
        inherit_context=True,
        outputs=[
            OutputMapping(child="import_artifacts.value", parent="import_artifacts.result"),
            OutputMapping(child="import_artifacts.marker", parent="import_artifacts.marker"),
        ],
        updates_context=True,
    )
    parent = Pipeline.from_step(import_step)

    runner = Flujo(parent, context_model=PipelineContext)
    res = await gather_result(runner, "ignored")
    ctx = res.final_pipeline_context
    assert isinstance(ctx, PipelineContext)

    # The key assertion: explicit None from child context should be preserved,
    # NOT replaced by "from_output" from the last step's output
    assert "result" in ctx.import_artifacts, "result key should exist even if value is None"
    assert ctx.import_artifacts.get("result") is None, (
        "Explicit None from context should be preserved, not replaced by output"
    )
    assert ctx.import_artifacts.get("marker") == "step1_ran"
