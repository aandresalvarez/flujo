from __future__ import annotations

import re
from typer.testing import CliRunner

from flujo.cli.main import app


def _clean(text: str) -> str:
    # Drop telemetry lines and trailing spaces; retain content for semantic checks
    ansi_re = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
    lines: list[str] = []
    for ln in text.splitlines():
        if re.match(r"\d{4}-\d{2}-\d{2}.*Logfire telemetry", ln):
            continue
        # Strip ANSI color codes and trailing whitespace
        ln = ansi_re.sub("", ln).rstrip()
        lines.append(ln)
    return "\n".join(lines)


def test_top_level_help_semantics() -> None:
    runner = CliRunner()
    out = _clean(runner.invoke(app, ["--help"], color=False).stdout)

    # Usage and description present
    assert "Usage: root" in out
    assert "A project-based server" in out

    # Options present (names and gist of help)
    assert "--profile" in out
    assert "--debug" in out and "no-debug" in out
    # Note: Rich tables may truncate long option names, so check for prefix
    assert "--install-complet" in out  # May be truncated as "--install-complet…"
    assert "--show-completion" in out
    assert "--help" in out

    # Commands present
    for cmd in ["init", "demo", "create", "run", "validate", "lens", "dev"]:
        assert cmd in out


def test_lens_help_semantics() -> None:
    runner = CliRunner()
    out = _clean(runner.invoke(app, ["lens", "--help"], color=False).stdout)

    # Usage and description
    assert "Usage: root lens" in out
    assert "Inspect" in out or "trace" in out

    # Commands under lens
    for cmd in ["list", "show", "trace", "replay", "spans", "stats"]:
        assert cmd in out


def test_dev_help_semantics() -> None:
    runner = CliRunner()
    out = _clean(runner.invoke(app, ["dev", "--help"], color=False).stdout)

    # Usage and description
    assert "Usage: root dev" in out
    assert "developer" in out or "diagnostic" in out

    # Commands under dev
    for cmd in [
        "version",
        "show-config",
        "explain",
        "validate",
        "compile-yaml",
        "visualize",
        "experimental",
        "budgets",
    ]:
        assert cmd in out
