from __future__ import annotations

from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml


def test_yaml_loader_instantiates_state_machine_step() -> None:
    yaml_text = """
version: "0.1"
steps:
  - kind: StateMachine
    name: SM
    start_state: s1
    end_states: [s2]
    states:
      s1:
        - kind: step
          name: A
      s2:
        - kind: step
          name: B
"""
    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    assert pipeline is not None
    assert len(pipeline.steps) == 1
    sm = pipeline.steps[0]
    from flujo.domain.dsl.state_machine import StateMachineStep

    assert isinstance(sm, StateMachineStep)
    assert set(sm.states.keys()) == {"s1", "s2"}
    assert sm.start_state == "s1"
