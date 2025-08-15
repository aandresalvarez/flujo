from __future__ import annotations

import re
from pathlib import Path
from typer.testing import CliRunner

from flujo.cli.main import app


def _normalize_help(text: str) -> str:
    # Drop telemetry/info log lines to keep snapshots stable
    lines = []
    for ln in text.splitlines():
        if re.match(r"\d{4}-\d{2}-\d{2}.*Logfire telemetry", ln):
            continue
        lines.append(ln)
    return "\n".join(lines).strip() + "\n"


def _read_snapshot(name: str) -> str:
    p = Path(__file__).with_name("snapshots") / name
    # Normalize trailing blank lines like the live output normalization
    return p.read_text().strip() + "\n"


def test_top_level_help_snapshot() -> None:
    runner = CliRunner()
    out = runner.invoke(app, ["--help"]).stdout
    assert _normalize_help(out) == _read_snapshot("top_help.txt")


def test_lens_help_snapshot() -> None:
    runner = CliRunner()
    out = runner.invoke(app, ["lens", "--help"]).stdout
    assert _normalize_help(out) == _read_snapshot("lens_help.txt")


def test_dev_help_snapshot() -> None:
    runner = CliRunner()
    out = runner.invoke(app, ["dev", "--help"]).stdout
    assert _normalize_help(out) == _read_snapshot("dev_help.txt")
