from __future__ import annotations

from pathlib import Path


import pytest


@pytest.mark.asyncio
async def test_loader_autoloads_skills_catalog_when_base_dir_is_provided(tmp_path: Path) -> None:
    # Write a skills.yaml in the temp project dir
    (tmp_path / "skills.yaml").write_text(
        """
echo_cmd:
  path: "tests.integration.skillmods.echo:make_echo"
  description: "Echo agent factory"
        """.strip()
    )

    # Write a minimal pipeline referencing the skill id
    p = tmp_path / "pipeline.yaml"
    p.write_text(
        """
version: "0.1"
steps:
  - name: Echo
    agent: { id: "echo_cmd", params: {} }
    input: "hello"
        """.strip()
    )

    from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml
    from flujo.application.runner import Flujo

    text = p.read_text()
    pipeline = load_pipeline_blueprint_from_yaml(text, base_dir=str(tmp_path))
    runner = Flujo(pipeline)
    result = await runner.run_async("")

    # The single step should have produced "hello" via the echo agent
    assert result.step_history, "no steps executed"
    assert result.step_history[-1].output == "hello"
