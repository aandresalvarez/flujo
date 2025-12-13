from __future__ import annotations

from pathlib import Path
import pytest

# StateMachine context manipulations are not xdist-safe
pytestmark = [pytest.mark.serial]


def _write_child_project(base: Path, name: str, tool_src: str, pipeline_yaml: str) -> Path:
    d = base / name
    (d / "skills").mkdir(parents=True, exist_ok=True)
    (d / "skills" / "__init__.py").write_text("# child skills package\n")
    (d / "skills" / "custom_tools.py").write_text(tool_src)
    (d / "pipeline.yaml").write_text(pipeline_yaml)
    return d


@pytest.mark.asyncio
async def test_regression_state_machine_import_step_no_missing_agent(tmp_path: Path) -> None:
    """Regression: importing a sub-pipeline inside a StateMachine state should not
    be treated as a plain Step (no agent) and should not raise MissingAgentError.
    Mirrors the original scenario name 'run_clarification_subpipeline'.
    """

    # Child pipeline emits import_artifacts.value = 1
    tools = (
        "from __future__ import annotations\n"
        "from typing import Any\n"
        "from flujo.domain.models import PipelineContext\n\n"
        "async def emit(_data: Any, *, context: PipelineContext | None = None) -> dict:\n"
        '    return {"import_artifacts": {"value": 1}}\n'
    )
    child_yaml = (
        'version: "0.1"\n'
        "steps:\n"
        "  - name: Emit\n"
        "    uses: skills.custom_tools:emit\n"
        "    updates_context: true\n"
    )
    child_dir = _write_child_project(tmp_path, "clarification", tools, child_yaml)

    # Parent StateMachine with an import step named like the original report
    parent_yaml = (
        'version: "0.1"\n'
        "imports:\n"
        f'  clar: "./{child_dir.name}/pipeline.yaml"\n'
        "steps:\n"
        "  - kind: StateMachine\n"
        "    name: Orchestrate\n"
        "    start_state: clarification\n"
        "    end_states: [done]\n"
        "    states:\n"
        "      clarification:\n"
        "        - name: run_clarification_subpipeline\n"
        "          uses: imports.clar\n"
        "          updates_context: true\n"
        "          config:\n"
        "            inherit_context: true\n"
        "            outputs:\n"
        '              - { child: "import_artifacts.value", parent: "import_artifacts.value" }\n'
        "      done:\n"
        "        - kind: step\n"
        "          name: Done\n"
    )

    from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml
    from flujo.application.runner import Flujo
    from flujo.domain.dsl.state_machine import StateMachineStep
    from flujo.domain.dsl.import_step import ImportStep

    import os

    os.environ["FLUJO_BLUEPRINT_ALLOWED_IMPORTS"] = "*"
    pipeline = load_pipeline_blueprint_from_yaml(parent_yaml, base_dir=str(tmp_path))
    # Ensure ImportStep compiled inside the state
    sm = pipeline.steps[0]
    assert isinstance(sm, StateMachineStep)
    s1 = sm.states.get("clarification")
    assert s1 is not None
    assert isinstance(s1.steps[0], ImportStep)

    # Run and ensure the value merged; no MissingAgentError surfaced
    runner = Flujo(pipeline)
    res = await runner.run_async("")
    ctx = res.final_pipeline_context
    assert ctx is not None
    assert ctx.import_artifacts.get("value") == 1
