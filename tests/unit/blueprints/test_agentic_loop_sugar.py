from __future__ import annotations

from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml


def test_agentic_loop_yaml_sugar_compiles() -> None:
    yaml_text = """
version: "0.1"
steps:
  - kind: agentic_loop
    name: al
    planner: "flujo.agents.recipes:NoOpReflectionAgent"
    registry: {}
"""
    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    assert len(pipeline.steps) == 1
    assert pipeline.steps[0].name == "al"
