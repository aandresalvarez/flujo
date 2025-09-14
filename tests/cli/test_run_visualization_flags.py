from __future__ import annotations

from pathlib import Path
import textwrap
from typer.testing import CliRunner

from flujo.cli.main import app


def _write_pipeline(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "pipe.py"
    p.write_text(textwrap.dedent(content))
    return p


def test_run_summary_hides_steps_and_context(tmp_path: Path) -> None:
    file = _write_pipeline(
        tmp_path,
        """
        from flujo.domain.dsl import Step, Pipeline

        async def s1(x: str) -> str:
            return x

        async def s2(x: str) -> dict:
            return {"echo": x}

        pipeline = Pipeline.from_step(Step.from_callable(s1, name="s1")) >> Step.from_callable(s2, name="s2")
        """,
    )
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(file), "--input", "hi", "--summary"])
    assert result.exit_code == 0
    out = result.stdout
    assert "Final output:" in out
    assert "Steps executed:" in out
    assert "Step Results:" not in out
    assert "Final Context:" not in out


def test_run_no_output_column_and_preview_len(tmp_path: Path) -> None:
    file = _write_pipeline(
        tmp_path,
        """
        from flujo.domain.dsl import Step, Pipeline

        async def long_out(_: str) -> str:
            return "A" * 200

        pipeline = Pipeline.from_step(Step.from_callable(long_out, name="s1"))
        """,
    )
    runner = CliRunner()
    # Hide context and trim output column preview; ensure the Output column header is hidden
    result = runner.invoke(
        app,
        [
            "run",
            str(file),
            "--input",
            "x",
            "--no-context",
            "--output-preview-len",
            "10",
            "--no-output-column",
        ],
    )
    assert result.exit_code == 0
    out = result.stdout
    assert "Step Results:" in out
    # Header should not include the 'Output' column when disabled (capitalized header)
    assert " Output " not in out


def test_run_final_output_format_json(tmp_path: Path) -> None:
    file = _write_pipeline(
        tmp_path,
        """
        from flujo.domain.dsl import Step, Pipeline

        async def make_obj(x: str) -> dict:
            return {"msg": x, "n": 1}

        pipeline = Pipeline.from_step(Step.from_callable(make_obj, name="obj"))
        """,
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            str(file),
            "--input",
            "hello",
            "--final-output-format",
            "json",
        ],
    )
    assert result.exit_code == 0
    out = result.stdout
    assert "Final output:" in out
    assert '{\n  "msg": "hello",\n  "n": 1\n}' in out


def test_run_only_steps_filters_table(tmp_path: Path) -> None:
    file = _write_pipeline(
        tmp_path,
        """
        from flujo.domain.dsl import Step, Pipeline

        async def alpha(x: str) -> str: return x
        async def bravo(x: str) -> str: return x
        async def charlie(x: str) -> str: return x

        p = Pipeline.from_step(Step.from_callable(alpha, name="alpha"))
        p = p >> Step.from_callable(bravo, name="bravo")
        pipeline = p >> Step.from_callable(charlie, name="charlie")
        """,
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            str(file),
            "--input",
            "z",
            "--only-steps",
            "bravo",
            "--no-context",
        ],
    )
    assert result.exit_code == 0
    out = result.stdout
    assert "Step Results:" in out
    assert "bravo" in out
    assert "alpha" not in out
    assert "charlie" not in out


def test_run_live_progress_prints_panels(tmp_path: Path) -> None:
    file = _write_pipeline(
        tmp_path,
        """
        from flujo.domain.dsl import Step, Pipeline

        async def s1(x: str) -> str: return x
        async def s2(x: str) -> str: return x

        pipeline = Pipeline.from_step(Step.from_callable(s1, name="s1")) >> Step.from_callable(s2, name="s2")
        """,
    )
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(file), "--input", "foo", "--live", "--no-context"])
    assert result.exit_code == 0
    out = result.stdout
    # ConsoleTracer panels include these titles/labels
    assert "Pipeline Start" in out
    assert "Step Start: s1" in out
    assert "Step End: s1" in out


def test_validate_plain_message_no_markup(tmp_path: Path) -> None:
    file = _write_pipeline(
        tmp_path,
        """
        from flujo.domain.dsl import Step, Pipeline
        async def s1(x: str) -> str: return x
        pipeline = Pipeline.from_step(Step.from_callable(s1, name="s1"))
        """,
    )
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(file)])
    assert result.exit_code == 0
    out = result.stdout
    assert "Pipeline is valid" in out
    assert "[green]" not in out
