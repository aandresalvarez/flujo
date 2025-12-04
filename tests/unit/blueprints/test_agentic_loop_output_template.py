from __future__ import annotations

import asyncio

from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml


def test_agentic_loop_output_template_wraps_mapper() -> None:
    # Ensure an event loop exists for components that create asyncio.Lock() during init
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create one for component initialization
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
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
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    else:
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
