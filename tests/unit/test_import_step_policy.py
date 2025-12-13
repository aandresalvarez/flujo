from __future__ import annotations

import pytest
from typing import Optional
from flujo.domain.models import PipelineContext


@pytest.mark.asyncio
async def test_import_step_projects_input_and_merges_context() -> None:
    from flujo.domain.dsl.step import Step
    from flujo.domain.dsl.pipeline import Pipeline
    from flujo.application.runner import Flujo
    from flujo.domain.dsl.import_step import ImportStep, OutputMapping
    from flujo.testing.utils import gather_result

    # Child pipeline: consumes cohort_definition/concept_sets from import_artifacts
    # and writes final_sql into import_artifacts
    async def build_query(_data: object, *, context: Optional[PipelineContext] = None) -> dict:
        assert context is not None
        cd = context.import_artifacts.get("cohort_definition")
        cs = context.import_artifacts.get("concept_sets") or []
        # Ensure values were projected without prompting
        assert cd is not None
        final_sql = f"-- cohorts: {str(cd)}; concepts: {len(cs)}"
        return {"import_artifacts": {"final_sql": final_sql}}

    child = Pipeline.from_step(
        Step.from_callable(build_query, name="query_builder", updates_context=True)
    )

    # Parent: Import the child pipeline via ImportStep with import_artifacts input projection
    import_step = ImportStep(
        name="run_qb",
        pipeline=child,
        inherit_context=True,
        input_to="import_artifacts",
        outputs=[
            OutputMapping(child="import_artifacts.final_sql", parent="import_artifacts.final_sql")
        ],
        updates_context=True,
    )
    parent = Pipeline.from_step(import_step)

    runner = Flujo(parent, context_model=PipelineContext)
    payload = {"cohort_definition": {"name": "demo"}, "concept_sets": [1, 2, 3]}
    res = await gather_result(runner, payload, initial_context_data={"initial_prompt": "goal"})
    ctx = res.final_pipeline_context
    assert isinstance(ctx, PipelineContext)
    # Ensure mapped value is available without re-prompt
    assert "final_sql" in ctx.import_artifacts
    assert str(ctx.import_artifacts["final_sql"]).startswith("-- cohorts:")
