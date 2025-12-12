from __future__ import annotations

from typer.testing import CliRunner
from pathlib import Path

from flujo.cli.main import app as app


def _write_yaml(path: Path) -> None:
    path.write_text(
        """
version: "0.1"
name: minimal
agents:
  echoer:
    id: "tests.unit.test_error_messages.need_str"
    model: "local:mock"
    system_prompt: "typed"
    output_schema:
      type: string
steps:
  - name: ask
    uses: agents.echoer
    input: "What is your name?"
  - name: run
    uses: agents.echoer
    input_schema:
      type: string
        """.strip()
    )


def test_credentials_hint_on_auth_exception(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create a tiny python pipeline that raises an auth-looking error
        py = Path("pipe.py")
        py.write_text(
            """
from flujo.domain.dsl import Step, Pipeline

class BoomAgent:
    async def run(self, data=None, **kwargs):
        raise Exception("401 Unauthorized: missing API key")

pipeline = Pipeline.from_step(Step(name="boom", agent=BoomAgent()))
            """.strip()
        )

        # Ensure we don't have OPENAI_API_KEY to trigger hint
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        res = runner.invoke(app, ["run", str(py)])
        out = res.stdout
        # Expect our credentials hint one-liner (specific env names may vary; check key phrase)
        assert "Credentials hint:" in out  # noqa: S101


def test_zero_config_non_interactive_dry_run_succeeds(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        yaml_path = Path("pipeline.yaml")
        _write_yaml(yaml_path)
        # Non-interactive JSON mode should not prompt, and should succeed without flujo.toml
        res = runner.invoke(app, ["run", str(yaml_path), "--dry-run", "--json"])
        assert res.exit_code == 0  # noqa: S101
        # Output should be JSON with validated:true
        out = res.stdout.strip()
        assert "validated" in out  # noqa: S101
