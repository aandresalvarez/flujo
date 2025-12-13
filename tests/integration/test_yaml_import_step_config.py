from __future__ import annotations

from textwrap import dedent

import pytest
from unittest import mock
from flujo.infra.config_manager import FlujoConfig


@pytest.fixture(autouse=True)
def mock_allowed_imports():
    """Allow test modules to be imported during blueprint loading."""
    with mock.patch("flujo.domain.blueprint.loader_resolution.get_config_provider") as mock_get:
        mock_config = mock.Mock(spec=FlujoConfig)
        mock_config.blueprint_allowed_imports = ["skills"]
        mock_config.settings = mock.Mock()
        mock_config.settings.blueprint_allowed_imports = ["skills"]

        mock_get.return_value.load_config.return_value = mock_config
        yield


@pytest.mark.asyncio
async def test_yaml_import_step_with_config(tmp_path, monkeypatch):
    from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml
    from flujo.application.runner import Flujo
    from flujo.domain.models import PipelineContext
    from flujo.testing.utils import gather_result

    # Create a simple skill that writes final_sql into import_artifacts using context
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
                cd = context.import_artifacts.get("cohort_definition")
                cs = context.import_artifacts.get("concept_sets") or []
                final_sql = f"-- cohorts: {str(cd)}; concepts: {len(cs)}"
                return {"import_artifacts": {"final_sql": final_sql}}
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
              input_to: import_artifacts
              outputs:
                - child: import_artifacts.final_sql
                  parent: import_artifacts.final_sql
        """
    )

    pipeline = load_pipeline_blueprint_from_yaml(parent_yaml, base_dir=str(tmp_path))
    runner = Flujo(pipeline, context_model=PipelineContext)
    payload = {"cohort_definition": {"name": "demo"}, "concept_sets": [1, 2, 3]}
    res = await gather_result(runner, payload, initial_context_data={"initial_prompt": "goal"})
    ctx = res.final_pipeline_context
    assert "final_sql" in ctx.import_artifacts
    assert str(ctx.import_artifacts["final_sql"]).startswith("-- cohorts:")
