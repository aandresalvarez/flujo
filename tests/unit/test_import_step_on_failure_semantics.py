from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_import_step_on_failure_skip_merges_nothing_and_succeeds() -> None:
    from flujo.domain.dsl.step import Step
    from flujo.domain.dsl.pipeline import Pipeline
    from flujo.application.runner import Flujo
    from flujo.domain.models import PipelineContext
    from flujo.testing.utils import gather_result
    from flujo.domain.dsl.import_step import ImportStep

    async def child_fails(_data: object) -> dict:  # no context needed
        raise RuntimeError("boom")

    child = Pipeline.from_step(Step.from_callable(child_fails, name="child"))

    import_step = ImportStep(
        name="run_child",
        pipeline=child,
        inherit_context=True,
        outputs=[],  # ensure no merge attempt
        on_failure="skip",
        updates_context=True,
    )
    parent = Pipeline.from_step(import_step)

    runner = Flujo(parent, context_model=PipelineContext)
    res = await gather_result(runner, "ignored")
    # Parent step should be marked success despite inner failure due to on_failure=skip
    assert res.step_history[-1].success is True
    # Context should remain unchanged (no merge)
    assert "boom" not in str(res.step_history[-1].feedback or "")
