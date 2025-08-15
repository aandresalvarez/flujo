from __future__ import annotations

import textwrap
from pathlib import Path
from typer.testing import CliRunner

from flujo.cli.main import app


def test_budgets_show_with_pipeline_entry(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Write flujo.toml with a specific pipeline budget
        Path("flujo.toml").write_text(
            textwrap.dedent(
                """
                [budgets]
                [budgets.pipeline]
                "my-pipe" = { total_cost_usd_limit = 3.5, total_tokens_limit = 123 }
                """
            ).strip()
        )

        result = runner.invoke(app, ["dev", "budgets", "show", "my-pipe"])
        assert result.exit_code == 0
        out = result.stdout
        assert "Effective budget for 'my-pipe':" in out
        assert "$3.50" in out
        assert "123" in out
        assert "Resolved from budgets.pipeline[my-pipe]" in out


def test_budgets_show_without_config(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["dev", "budgets", "show", "any-pipe"])
        assert result.exit_code == 0
        assert "No budget configured (unlimited). Source: none" in result.stdout


def test_pipeline_mermaid_outputs_code_fence(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pipe = Path("pipe.py")
        pipe.write_text(
            textwrap.dedent(
                """
                from flujo.domain.dsl import Step, Pipeline

                async def a(x: str) -> str: return x
                async def b(x: str) -> str: return x

                s1 = Step.from_callable(a, name="a")
                s2 = Step.from_callable(b, name="b")
                pipeline = Pipeline.from_step(s1) >> s2
                """
            )
        )

        result = runner.invoke(
            app,
            [
                "dev",
                "visualize",
                "--file",
                str(pipe),
                "--object",
                "pipeline",
                "--detail-level",
                "low",
            ],
        )
        assert result.exit_code == 0
        assert "```mermaid" in result.stdout


def test_version_and_show_config(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        res_v = runner.invoke(app, ["dev", "version"])
        assert res_v.exit_code == 0
        assert "flujo version:" in res_v.stdout

        res_cfg = runner.invoke(app, ["dev", "show-config"])
        assert res_cfg.exit_code == 0
        # Should print a dict-like structure without secrets
        assert "{" in res_cfg.stdout and "}" in res_cfg.stdout
        assert "openai_api_key" not in res_cfg.stdout
        assert "logfire_api_key" not in res_cfg.stdout
