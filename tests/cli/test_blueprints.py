from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typer.testing import CliRunner
from flujo.cli.main import app


def test_cli_budgets_show_works(tmp_path: Path) -> None:
    # Create a minimal flujo.toml in temp dir
    toml = tmp_path / "flujo.toml"
    toml.write_text(
        """
[budgets.default]
 total_cost_usd_limit = 10.0
 total_tokens_limit = 1000

 [budgets.pipeline]
 "team-*" = { total_cost_usd_limit = 5.0 }
        """.strip()
    )

    # Run CLI in that directory so config is discovered
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "dev",
            "budgets",
            "show",
            "team-alpha",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    out = result.stdout + result.stderr
    assert "Effective budget for 'team-alpha'" in out
    assert "Resolved from budgets.pipeline[team-*]" in out


runner = CliRunner()

YAML_FULL = """
version: "0.1"
agents:
  categorizer:
    id: "flujo.builtins.stringify"
    model: "openai:gpt-4o"
    system_prompt: "categorize"
    output_schema: { type: string }
steps:
  - kind: step
    name: categorize
    uses: agents.categorizer
    input: "hi"
  - kind: step
    name: subpipe
    uses: imports.support
    input: "{{ previous_step.output }}"
"""


def test_run_fully_declarative_yaml(tmp_path, monkeypatch):
    # Monkeypatch agent factory to avoid real API keys/network
    def _fake_make_agent_async(*args: object, **kwargs: object):
        async def _agent(x: object, *_a: object, **_k: object) -> object:
            return "a"

        return _agent

    monkeypatch.setattr(
        "flujo.domain.blueprint.compiler.make_agent_async",
        _fake_make_agent_async,
        raising=True,
    )

    # Write a simple imported sub-pipeline
    sub = tmp_path / "support.yaml"
    sub.write_text("version: '0.1'\nsteps:\n  - kind: step\n    name: inner\n")

    p = tmp_path / "workflow.yaml"
    yaml_text = """
version: "0.1"
imports:
  support: "./support.yaml"
agents:
  categorizer:
    id: "tests.unit.test_error_messages.need_str"
    model: "local:mock"
    system_prompt: "typed"
    output_schema: { type: string }
steps:
  - kind: step
    name: categorize
    uses: agents.categorizer
    input: "hi"
  - kind: step
    name: use_support
    uses: imports.support
    meta:
      is_adapter: true
      adapter_id: generic-adapter
      adapter_allow: generic
    input_schema: { type: string }
    input: "{{ previous_step.output }}"
"""
    p.write_text(yaml_text)
    result = runner.invoke(app, ["run", str(p), "--input", "hello", "--json"])
    # With monkeypatched agent, execution should succeed without external API keys
    assert result.exit_code == 0, result.output


def test_yaml_imports_and_templated_input(tmp_path):
    # Write imported support workflow (simple passthrough step)
    support = tmp_path / "support_workflow.yaml"
    support.write_text(
        """
version: "0.1"
steps:
  - kind: step
    name: inner_support
    agent:
      id: "tests.unit.test_error_messages.need_str"
      model: "local:mock"
      system_prompt: "typed"
    output_schema:
      type: string
    output_schema:
      type: string
        """.strip()
    )

    # Write onboarding workflow that imports support
    onboarding = tmp_path / "onboarding_workflow.yaml"
    onboarding.write_text(
        f"""
version: "0.1"
imports:
  support: "./{support.name}"
steps:
  - kind: step
    name: greet
    agent:
      id: "tests.unit.test_error_messages.need_str"
      model: "local:mock"
      system_prompt: "typed"
    meta:
      is_adapter: true
      adapter_id: generic-adapter
      adapter_allow: generic
    output_schema: {{ type: string }}
  - kind: step
    name: support_assist
    uses: imports.support
    meta:
      is_adapter: true
      adapter_id: generic-adapter
      adapter_allow: generic
    input_schema: {{ type: string }}
        """.strip()
    )

    # Run CLI
    local_runner = CliRunner()
    result = local_runner.invoke(
        app,
        [
            "run",
            str(onboarding),
            "--input",
            "Welcome!",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    # Basic test that the pipeline runs successfully with imports
    assert "greet" in result.output
    assert "support_assist" in result.output
