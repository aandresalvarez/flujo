from __future__ import annotations

from flujo.domain.dsl import Pipeline


def _validate(yaml_text: str):
    p = Pipeline.from_yaml_text(yaml_text)
    return p.validate_graph()


def test_va5_suppressed_when_input_references_previous_by_name() -> None:
    yaml_text = """
version: "0.1"
name: "va5_by_name"
steps:
  - name: a
    agent: "flujo.builtins.passthrough"
    input: "Hello"
  - name: b
    agent: "flujo.builtins.passthrough"
    input: "{{ steps.a.output }}"
"""
    report = _validate(yaml_text)
    assert not any(f.rule_id == "V-A5" for f in report.warnings)


def test_va5_suppressed_when_input_references_previous_keyword() -> None:
    yaml_text = """
version: "0.1"
name: "va5_previous_step"
steps:
  - name: a
    agent: "flujo.builtins.passthrough"
    input: "Hello"
  - name: b
    agent: "flujo.builtins.passthrough"
    input: "{{ previous_step }}"
"""
    report = _validate(yaml_text)
    assert not any(f.rule_id == "V-A5" for f in report.warnings)


def test_va5_warns_when_no_template_consumes_previous() -> None:
    yaml_text = """
version: "0.1"
name: "va5_no_reference"
steps:
  - name: a
    agent: "flujo.builtins.passthrough"
    input: "Hello"
  - name: b
    agent: "flujo.builtins.passthrough"
    input: "Static"
"""
    report = _validate(yaml_text)
    assert any(f.rule_id == "V-A5" for f in report.warnings)
