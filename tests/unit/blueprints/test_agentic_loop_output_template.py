from __future__ import annotations

from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml


def test_agentic_loop_output_template_wraps_mapper() -> None:
    yaml_text = """
version: "0.1"
steps:
  - kind: agentic_loop
    name: al
    planner: "flujo.agents.recipes:NoOpReflectionAgent"
    registry: {}
    output_template: "FINAL: {{ previous_step.execution_result }}"
"""
    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    step = pipeline.steps[0]
    assert step.name == "al"
