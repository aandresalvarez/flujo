from __future__ import annotations

import textwrap

from flujo.domain.dsl.pipeline_validation_helpers import apply_fallback_template_lints
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.pipeline_validation import ValidationReport


def _load_pipeline(yaml_text: str) -> Pipeline:
    return Pipeline.from_yaml_text(textwrap.dedent(yaml_text))


def test_yaml_pipeline_generic_requires_adapter() -> None:
    yaml_text = """
    steps:
      - name: a
        agent: tests.unit.test_error_messages.make_int
      - name: b
        agent: tests.unit.test_error_messages.need_str
    """
    p = _load_pipeline(yaml_text)
    report = p.validate_graph()
    assert any(f.rule_id == "V-A2-STRICT" for f in report.errors)


def test_yaml_adapter_must_have_allowlist_token() -> None:
    yaml_text = """
    steps:
      - name: a
        agent: tests.unit.test_error_messages.make_int
      - name: adapt
        agent: tests.unit.test_error_messages.make_int
        meta:
          is_adapter: true
          adapter_id: generic-adapter
          adapter_allow: wrong
    """
    p = _load_pipeline(yaml_text)
    report = ValidationReport()
    apply_fallback_template_lints(p, report)
    report2 = p.validate_graph()
    all_errors = report.errors + report2.errors
    assert any(f.rule_id == "V-ADAPT-ALLOW" for f in all_errors)
