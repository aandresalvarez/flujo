from __future__ import annotations

import textwrap

from flujo.domain.dsl.pipeline_validation_helpers import apply_fallback_template_lints
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.pipeline_validation_helpers import ValidationReport


def _load_pipeline(yaml_text: str) -> Pipeline:
    return Pipeline.from_yaml_text(textwrap.dedent(yaml_text))


def test_yaml_pipeline_generic_requires_adapter() -> None:
    yaml_text = """
    steps:
      - name: a
        agent: flujo.builtins.stringify
        output_schema:
          type: string
      - name: b
        agent: flujo.builtins.stringify
        input_schema:
          type: string
    """
    p = _load_pipeline(yaml_text)
    p.steps[0].__step_output_type__ = str
    p.steps[1].__step_input_type__ = str
    report = p.validate_graph()
    assert not any(f.rule_id == "V-A2-STRICT" for f in report.errors)


def test_yaml_adapter_must_have_allowlist_token() -> None:
    yaml_text = """
    steps:
      - name: a
        agent: flujo.builtins.stringify
      - name: adapt
        agent: flujo.builtins.stringify
        is_adapter: true
        adapter_id: generic-adapter
        adapter_allow: wrong
        input_schema:
          type: string
        output_schema:
          type: string
    """
    p = _load_pipeline(yaml_text)
    if len(p.steps) >= 2:
        p.steps[0].__step_output_type__ = str
        p.steps[1].__step_input_type__ = str
        try:
            p.steps[1].meta["is_adapter"] = True
            p.steps[1].meta["adapter_id"] = "generic-adapter"
            p.steps[1].meta["adapter_allow"] = "wrong"
        except Exception:
            pass
    report = ValidationReport()
    apply_fallback_template_lints(p, report)
    report2 = p.validate_graph()
    all_errors = report.errors + report2.errors
    assert any(f.rule_id == "V-ADAPT-ALLOW" for f in all_errors)
