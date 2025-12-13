from __future__ import annotations

from typing import Any

import pytest

from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml, BlueprintError
from flujo.domain.dsl.state_machine import StateMachineStep
from flujo.domain.dsl.pipeline import Pipeline


def _build_yaml(transitions: str) -> str:
    return (
        "steps:\n"
        "  - kind: StateMachine\n"
        "    name: orchestrate\n"
        "    start_state: s1\n"
        "    end_states: [s3]\n"
        "    states:\n"
        "      s1:\n"
        "        steps:\n"
        "          - name: noop1\n"
        "            kind: step\n"
        "            agent: null\n"
        "      s2:\n"
        "        steps:\n"
        "          - name: noop2\n"
        "            kind: step\n"
        "            agent: null\n"
        "      s3:\n"
        "        steps: []\n"
        f"    transitions:\n{transitions}"
    )


def test_transitions_parse_and_validate_success() -> None:
    yaml_text = _build_yaml(
        "      - from: s1\n"
        "        on: success\n"
        "        to: s2\n"
        '      - from: "*"\n'
        "        on: failure\n"
        "        to: s3\n"
    )
    pipe: Pipeline[Any, Any] = load_pipeline_blueprint_from_yaml(yaml_text)
    assert isinstance(pipe, Pipeline)
    assert pipe.steps, "pipeline must contain steps"
    sm = pipe.steps[0]
    assert isinstance(sm, StateMachineStep)
    assert len(sm.transitions) == 2
    # from alias should bind to from_state
    assert sm.transitions[0].from_state == "s1"
    assert sm.transitions[0].on == "success"
    assert sm.transitions[0].to == "s2"


def test_transitions_invalid_to_state_raises() -> None:
    yaml_text = _build_yaml("      - from: s1\n        on: success\n        to: not_a_state\n")
    with pytest.raises(BlueprintError):
        load_pipeline_blueprint_from_yaml(yaml_text)


def test_transitions_wildcard_from_ok() -> None:
    yaml_text = _build_yaml('      - from: "*"\n        on: success\n        to: s3\n')
    pipe: Pipeline[Any, Any] = load_pipeline_blueprint_from_yaml(yaml_text)
    sm = pipe.steps[0]
    assert isinstance(sm, StateMachineStep)
    assert sm.transitions[0].from_state == "*"


def test_transitions_invalid_on_raises() -> None:
    yaml_text = _build_yaml("      - from: s1\n        on: bogus\n        to: s3\n")
    with pytest.raises(BlueprintError):
        load_pipeline_blueprint_from_yaml(yaml_text)


def test_transitions_when_expression_parses_at_load_time() -> None:
    yaml_text = _build_yaml(
        "      - from: s1\n"
        "        on: success\n"
        "        to: s2\n"
        "        when: \"context.import_artifacts.get('x')\"\n"
    )
    pipe: Pipeline[Any, Any] = load_pipeline_blueprint_from_yaml(yaml_text)
    sm = pipe.steps[0]
    assert isinstance(sm, StateMachineStep)
    assert sm.transitions and sm.transitions[0].when == "context.import_artifacts.get('x')"
