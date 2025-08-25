from __future__ import annotations

import flujo.builtins  # noqa: F401  # ensure built-in skills are registered for skill resolution
from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml
from flujo.application.runner import Flujo

YAML_SPEC = """
version: "0.1"
name: "template_fallbacks_bug_test"

steps:
  - kind: step
    name: test_fallback
    agent:
      id: "flujo.builtins.stringify"
    input: "{{ context.initial_prompt or 'Fallback: No prompt provided' }}"

  - kind: step
    name: show_result
    agent:
      id: "flujo.builtins.stringify"
    input: "Template result: {{ steps.test_fallback }}"
"""


def test_yaml_template_fallbacks_end_to_end() -> None:
    pipeline = load_pipeline_blueprint_from_yaml(YAML_SPEC)
    runner = Flujo(pipeline=pipeline)
    # Empty initial input; the first step should use the fallback literal
    result = runner.run("")
    # Last step output is the final pipeline output
    last = result.step_history[-1]
    assert last.output == "Template result: Fallback: No prompt provided"
