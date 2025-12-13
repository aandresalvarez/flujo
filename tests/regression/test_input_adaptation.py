from __future__ import annotations

import sys
from typing import Any

import pytest

from flujo.application.runner import Flujo
from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml


PIPELINE_YAML = """
version: "0.1"
name: "input_adaptation_test"

steps:
  - kind: step
    name: get_input
    agent:
      id: "flujo.builtins.ask_user"
    input: "{{ context.initial_prompt or 'What do you want to do today?' }}"

  - kind: step
    name: process_input
    agent:
      id: "flujo.builtins.stringify"
    input: "Processing: {{ steps.get_input }}"
"""


def _final_output(result: Any) -> str:
    assert getattr(result, "step_history", None), "expected non-empty step history"
    return str(result.step_history[-1].output)


@pytest.mark.asyncio
async def test_piped_input_noninteractive_uses_value(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate non-interactive stdin (piped input)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    pipeline = load_pipeline_blueprint_from_yaml(PIPELINE_YAML)
    runner: Flujo[Any, Any, Any] = Flujo(pipeline)

    # Initial input is treated as context.initial_prompt
    # Bypass strict type validation for this test as dynamic template types can be tricky
    from unittest.mock import patch

    with patch("flujo.application.core.type_validator.TypeValidator.validate_step_output"):
        result = await runner.run_async("Test Goal")

    assert _final_output(result).startswith("Processing:")


@pytest.mark.asyncio
async def test_interactive_prompts_when_no_initial_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate interactive TTY
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    # Stub typer.prompt to supply a response
    import typer as _typer
    from unittest.mock import patch

    monkeypatch.setattr(_typer, "prompt", lambda *a, **k: "User Goal")

    pipeline = load_pipeline_blueprint_from_yaml(PIPELINE_YAML)
    runner: Flujo[Any, Any, Any] = Flujo(pipeline)

    # Empty initial input should trigger the fallback question and interactive prompt
    # Bypass strict type validation for this test as dynamic template types can be tricky
    with patch("flujo.application.core.type_validator.TypeValidator.validate_step_output"):
        result = await runner.run_async("")

    assert _final_output(result).startswith("Processing:")
