from __future__ import annotations

from pathlib import Path
from unittest import mock
import pytest
from flujo.infra.config_manager import FlujoConfig


@pytest.fixture(autouse=True)
def mock_allowed_imports():
    """Allow test modules to be imported during blueprint loading."""
    with mock.patch("flujo.domain.blueprint.loader_resolution.get_config_provider") as mock_get:
        mock_config = mock.Mock(spec=FlujoConfig)
        mock_config.blueprint_allowed_imports = ["skills", "imports"]
        mock_config.settings = mock.Mock()
        mock_config.settings.blueprint_allowed_imports = ["skills", "imports"]

        mock_get.return_value.load_config.return_value = mock_config
        yield


def _write_child_project(base: Path, name: str, tool_src: str, pipeline_yaml: str) -> Path:
    d = base / name
    (d / "skills").mkdir(parents=True, exist_ok=True)
    (d / "skills" / "__init__.py").write_text("# child skills package\n")
    (d / "skills" / "custom_tools.py").write_text(tool_src)
    (d / "pipeline.yaml").write_text(pipeline_yaml)
    return d


def test_statemachine_state_allows_inline_steps_mapping(tmp_path: Path) -> None:
    # Child emits import_artifacts.k = v
    tools = (
        "from __future__ import annotations\n"
        "from typing import Any, Optional\n"
        "from flujo.domain.models import PipelineContext\n\n"
        "async def emit(_data: Any, *, context: Optional[PipelineContext] = None) -> dict:\n"
        '    return {"import_artifacts": {"k": "v"}}\n'
    )
    child_yaml = (
        'version: "0.1"\n'
        "steps:\n"
        "  - name: Emit\n"
        "    uses: skills.custom_tools:emit\n"
        "    updates_context: true\n"
    )
    child_dir = _write_child_project(tmp_path, "child", tools, child_yaml)

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
        "        steps:\n"
        "          - name: ImportChild\n"
        "            uses: imports.c\n"
        "            updates_context: true\n"
        "            config:\n"
        "              inherit_context: true\n"
        "              outputs:\n"
        '                - { child: "import_artifacts.k", parent: "import_artifacts.k" }\n'
        "      done:\n"
        "        steps:\n"
        "          - kind: step\n"
        "            name: Done\n"
    )

    from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml
    from flujo.domain.dsl.state_machine import StateMachineStep
    from flujo.domain.dsl.import_step import ImportStep

    pipeline = load_pipeline_blueprint_from_yaml(parent_yaml, base_dir=str(tmp_path))
    sm = pipeline.steps[0]
    assert isinstance(sm, StateMachineStep)
    s1 = sm.states.get("s1")
    assert s1 is not None
    assert len(s1.steps) == 1
    assert isinstance(s1.steps[0], ImportStep)


def test_conditional_branch_allows_inline_steps_mapping() -> None:
    yaml_text = (
        'version: "0.1"\n'
        "steps:\n"
        "  - kind: conditional\n"
        "    name: C\n"
        "    condition_expression: 'true'\n"
        "    branches:\n"
        "      true:\n"
        "        steps:\n"
        "          - kind: step\n"
        "            name: A\n"
        "      false:\n"
        "        steps:\n"
        "          - kind: step\n"
        "            name: B\n"
    )
    from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml

    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    assert pipeline is not None
    assert len(pipeline.steps) == 1
    cond = pipeline.steps[0]
    from flujo.domain.dsl.conditional import ConditionalStep

    assert isinstance(cond, ConditionalStep)
    assert set(cond.branches.keys()) == {True, False} or set(cond.branches.keys()) == {
        "true",
        "false",
    }
