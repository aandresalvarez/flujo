from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def _write_child_project(
    base: Path,
    name: str,
    custom_tools_src: str,
    pipeline_yaml: str,
) -> Path:
    d = base / name
    (d / "skills").mkdir(parents=True, exist_ok=True)
    (d / "skills" / "__init__.py").write_text("# child skills package\n")
    (d / "skills" / "custom_tools.py").write_text(custom_tools_src)
    (d / "pipeline.yaml").write_text(pipeline_yaml)
    return d


def test_imported_children_resolve_isolated_skills_and_merge_outputs(tmp_path: Path) -> None:
    """End-to-end check: three imported children, each with its own 'skills' package.

    Ensures that:
    - each child resolves its local skills module without PYTHONPATH hacks
    - sys.modules collisions do not cause cross-child bleed
    - outputs mapping merges only the specified fields into the parent
    """

    # Clarification child: writes cohort_definition
    clarify_tools = dedent(
        """
        from __future__ import annotations
        from typing import Any
        from flujo.domain.models import PipelineContext

        async def make_output(_data: Any, *, context: PipelineContext | None = None) -> dict:
            # emit a distinct value so we can detect module bleed
            return {"import_artifacts": {"cohort_definition": {"source": "clarification", "id": 1}}}
        """
    ).strip()
    clarify_yaml = dedent(
        """
        version: "0.1"
        steps:
          - name: Clarify
            uses: skills.custom_tools:make_output
            updates_context: true
        """
    ).strip()

    # Concept Discovery child: writes concept_sets
    concept_tools = dedent(
        """
        from __future__ import annotations
        from typing import Any
        from flujo.domain.models import PipelineContext

        async def make_output(_data: Any, *, context: PipelineContext | None = None) -> dict:
            return {"import_artifacts": {"concept_sets": ["cs-A", "cs-B"]}}
        """
    ).strip()
    concept_yaml = dedent(
        """
        version: "0.1"
        steps:
          - name: Discover Concepts
            uses: skills.custom_tools:make_output
            updates_context: true
        """
    ).strip()

    # Query Builder child: reads prior fields and writes final_sql
    query_tools = dedent(
        """
        from __future__ import annotations
        from typing import Any
        from flujo.domain.models import PipelineContext

        async def make_output(_data: Any, *, context: PipelineContext | None = None) -> dict:
            assert context is not None
            cd = context.import_artifacts.get("cohort_definition")
            cs = context.import_artifacts.get("concept_sets")
            # Build a final_sql that reflects both inputs to catch wrong skills usage
            return {
                "import_artifacts": {
                    "final_sql": f"-- cd:{cd['source'] if isinstance(cd, dict) else 'nil'}; cs:{len(cs or [])}"
                }
            }
        """
    ).strip()
    query_yaml = dedent(
        """
        version: "0.1"
        steps:
          - name: Build Query
            uses: skills.custom_tools:make_output
            updates_context: true
        """
    ).strip()

    # Layout: tmp/{main,clarification,concept_discovery,query_builder}
    main = tmp_path / "main"
    main.mkdir()
    _write_child_project(tmp_path, "clarification", clarify_tools, clarify_yaml)
    _write_child_project(tmp_path, "concept_discovery", concept_tools, concept_yaml)
    _write_child_project(tmp_path, "query_builder", query_tools, query_yaml)

    # Parent pipeline importing the three children and mapping outputs
    parent_yaml = dedent(
        """
        version: "0.1"
        imports:
          clarification: "../clarification/pipeline.yaml"
          concept_discovery: "../concept_discovery/pipeline.yaml"
          query_builder: "../query_builder/pipeline.yaml"
        steps:
          - name: Clarification
            uses: imports.clarification
            updates_context: true
            config:
              inherit_context: true
              outputs:
                - { child: "import_artifacts.cohort_definition", parent: "import_artifacts.cohort_definition" }
          - name: Concept Discovery
            uses: imports.concept_discovery
            updates_context: true
            config:
              inherit_context: true
              outputs:
                - { child: "import_artifacts.concept_sets", parent: "import_artifacts.concept_sets" }
          - name: Query Builder
            uses: imports.query_builder
            updates_context: true
            config:
              inherit_context: true
              outputs:
                - { child: "import_artifacts.final_sql", parent: "import_artifacts.final_sql" }
        """
    ).strip()

    (main / "pipeline.yaml").write_text(parent_yaml)

    # Load and run via public API using proper base_dir for parent
    from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml
    from flujo.application.runner import Flujo

    text = (main / "pipeline.yaml").read_text()
    pipeline = load_pipeline_blueprint_from_yaml(text, base_dir=str(main))
    runner = Flujo(pipeline)
    result = runner.run("")

    # Validate merged outputs are present and reflect each child module, not bled
    ctx = result.final_pipeline_context
    assert ctx is not None

    artifacts = getattr(ctx, "import_artifacts", {})
    # From clarification child
    cd = artifacts.get("cohort_definition") or {}
    assert isinstance(cd, dict) and cd.get("source") == "clarification"
    # From concept discovery child
    concepts = artifacts.get("concept_sets")
    assert concepts == ["cs-A", "cs-B"]
    # From query builder child; ensure it saw the above values
    final_sql = artifacts.get("final_sql") or ""
    assert str(final_sql).startswith("-- cd:clarification; cs:2")
