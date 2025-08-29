from __future__ import annotations

from textwrap import dedent

import pytest


@pytest.mark.asyncio
async def test_yaml_import_step_with_config(tmp_path, monkeypatch):
    from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml
    from flujo.application.runner import Flujo
    from flujo.domain.models import PipelineContext
    from flujo.testing.utils import gather_result

    # Create a simple skill that writes final_sql into scratchpad using context
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "__init__.py").write_text("")
    helpers_py = skills_dir / "helpers.py"
    helpers_py.write_text(
        dedent(
            """
            from __future__ import annotations
            from flujo.domain.models import PipelineContext

            async def emit_final_sql(_data: object, *, context: PipelineContext | None = None) -> dict:
                assert context is not None
                cd = context.scratchpad.get("cohort_definition")
                cs = context.scratchpad.get("concept_sets") or []
                final_sql = f"-- cohorts: {str(cd)}; concepts: {len(cs)}"
                return {"scratchpad": {"final_sql": final_sql}}
            """
        )
    )

    # Ensure tmp_path is importable
    monkeypatch.syspath_prepend(str(tmp_path))

    # Child pipeline YAML: single step that updates context
    child_yaml = dedent(
        """
        version: "0.1"
        steps:
          - kind: step
            name: qb
            uses: skills.helpers:emit_final_sql
            updates_context: true
        """
    )
    child_path = tmp_path / "child.yaml"
    child_path.write_text(child_yaml)

    # Parent YAML with imports and ImportStep config
    parent_yaml = dedent(
        f"""
        version: "0.1"
        imports:
          qb: "{child_path.name}"
        steps:
          - kind: step
            name: run_query_builder
            uses: imports.qb
            updates_context: true
            config:
              input_to: scratchpad
              outputs:
                - child: scratchpad.final_sql
                  parent: scratchpad.final_sql
        """
    )

    pipeline = load_pipeline_blueprint_from_yaml(parent_yaml, base_dir=str(tmp_path))
    runner = Flujo(pipeline, context_model=PipelineContext)
    payload = {"cohort_definition": {"name": "demo"}, "concept_sets": [1, 2, 3]}
    res = await gather_result(runner, payload, initial_context_data={"initial_prompt": "goal"})
    ctx = res.final_pipeline_context
    assert "final_sql" in ctx.scratchpad
    assert str(ctx.scratchpad["final_sql"]).startswith("-- cohorts:")
