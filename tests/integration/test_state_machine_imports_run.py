from __future__ import annotations

import pytest
from pathlib import Path

pytestmark = pytest.mark.serial


def _write_child_project(base: Path, name: str, tool_src: str, pipeline_yaml: str) -> Path:
    d = base / name
    (d / "skills").mkdir(parents=True, exist_ok=True)
    (d / "skills" / "__init__.py").write_text("# child skills package\n")
    (d / "skills" / "custom_tools.py").write_text(tool_src)
    (d / "pipeline.yaml").write_text(pipeline_yaml)
    return d


def test_state_machine_with_import_step_executes_and_merges_context(tmp_path: Path) -> None:
    # Child emits import_artifacts.foo = "bar"
    tools = (
        "from __future__ import annotations\n"
        "from typing import Any\n"
        "from flujo.domain.models import PipelineContext\n\n"
        "async def make_output(_data: Any, *, context: PipelineContext | None = None) -> dict:\n"
        '    return {"import_artifacts": {"foo": "bar"}}\n'
    )
    child_yaml = (
        'version: "0.1"\n'
        "steps:\n"
        "  - name: Child\n"
        "    uses: skills.custom_tools:make_output\n"
        "    updates_context: true\n"
    )
    child_dir = _write_child_project(tmp_path, "child", tools, child_yaml)

    # Parent with StateMachine that imports the child pipeline in state s1
    parent_yaml = (
        'version: "0.1"\n'
        "imports:\n"
        f'  c: "./{child_dir.name}/pipeline.yaml"\n'
        "steps:\n"
        "  - kind: StateMachine\n"
        "    name: SM\n"
        "    start_state: s1\n"
        "    end_states: [done]\n"
        "    states:\n"
        "      s1:\n"
        "        - name: ImportChild\n"
        "          uses: imports.c\n"
        "          updates_context: true\n"
        "          config:\n"
        "            inherit_context: true\n"
        "            outputs:\n"
        '              - { child: "import_artifacts.foo", parent: "import_artifacts.foo" }\n'
        "      done:\n"
        "        - kind: step\n"
        "          name: Done\n"
    )

    from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml
    from flujo.application.runner import Flujo

    import os

    os.environ["FLUJO_BLUEPRINT_ALLOWED_IMPORTS"] = "*"
    pipeline = load_pipeline_blueprint_from_yaml(parent_yaml, base_dir=str(tmp_path))
    runner = Flujo(pipeline)
    result = runner.run("")

    ctx = result.final_pipeline_context
    assert ctx is not None
    assert ctx.import_artifacts.get("foo") == "bar"
