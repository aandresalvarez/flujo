from __future__ import annotations

from pathlib import Path
from typer.testing import CliRunner

from flujo.cli.main import app


runner = CliRunner()


def test_cli_create_wizard_writes_natural_yaml(tmp_path: Path) -> None:
    out_dir = tmp_path / "proj"
    res = runner.invoke(
        app,
        [
            "create",
            "--wizard",
            "--non-interactive",
            "--goal",
            "Summarize latest news",
            "--name",
            "clarification_loop",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0, res.output
    p = out_dir / "pipeline.yaml"
    assert p.exists(), "pipeline.yaml not written by wizard"
    text = p.read_text()
    assert "conversation: true" in text
    assert "propagation: auto" in text or "next_input: previous_output" in text
    assert "stop_when: agent_finished" in text
    assert "output:" in text


def test_cli_explain_summarizes_yaml(tmp_path: Path) -> None:
    p = tmp_path / "pipe.yaml"
    p.write_text(
        """
version: "0.1"
steps:
  - kind: loop
    name: test
    loop:
      conversation: true
      propagation: auto
      output:
        text: conversation_history
      body:
        - kind: step
          name: inner_step
""".strip()
    )
    res = runner.invoke(app, ["dev", "explain", str(p)])
    assert res.exit_code == 0, res.output
    assert "test" in res.stdout


def test_cli_create_wizard_map_and_parallel(tmp_path: Path) -> None:
    out_dir1 = tmp_path / "proj_map"
    res1 = runner.invoke(
        app,
        [
            "create",
            "--wizard",
            "--wizard-pattern",
            "map",
            "--non-interactive",
            "--name",
            "map_demo",
            "--output-dir",
            str(out_dir1),
        ],
    )
    assert res1.exit_code == 0, res1.output
    t1 = (out_dir1 / "pipeline.yaml").read_text()
    assert "kind: map" in t1 and "iterable_input: items" in t1

    out_dir2 = tmp_path / "proj_par"
    res2 = runner.invoke(
        app,
        [
            "create",
            "--wizard",
            "--wizard-pattern",
            "parallel",
            "--wizard-reduce-mode",
            "concat",
            "--non-interactive",
            "--name",
            "par_demo",
            "--output-dir",
            str(out_dir2),
        ],
    )
    assert res2.exit_code == 0, res2.output
    t2 = (out_dir2 / "pipeline.yaml").read_text()
    assert "kind: parallel" in t2 and "reduce: concat" in t2
