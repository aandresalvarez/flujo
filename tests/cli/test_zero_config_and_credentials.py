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
    model: "openai:gpt-4o"  # used only for provider hint; no network call on --dry-run
    system_prompt: |
      You are a test agent.
    output_schema:
      type: object
      properties:
        msg: { type: string }
      required: [msg]
steps:
  - name: ask
    kind: hitl
    message: "What is your name?"
  - name: run
    uses: agents.echoer
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
