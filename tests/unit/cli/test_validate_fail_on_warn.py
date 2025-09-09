from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import typer

from flujo.cli.exit_codes import EX_VALIDATION_FAILED
from flujo.cli.main import _validate_impl


def _write_yaml(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "pipe.yaml"
    p.write_text(text)
    return p


def test_fail_on_warn_triggers_nonzero_exit(tmp_path: Path) -> None:
    # Deliberately trigger V-T1 (previous_step.output misuse)
    y = _write_yaml(
        tmp_path,
        textwrap.dedent(
            """
            version: "0.1"
            steps:
              - name: s1
                agent: { id: "flujo.builtins.stringify" }
                input: "hello"
                updates_context: true
              - name: s2
                agent: { id: "flujo.builtins.stringify" }
                input: "{{ previous_step.output }}"
            """
        ),
    )

    with pytest.raises(typer.Exit) as ei:
        _validate_impl(
            str(y),
            strict=False,
            output_format="json",
            include_imports=False,
            fail_on_warn=True,
            rules=None,
        )
    assert ei.value.exit_code == EX_VALIDATION_FAILED


def test_without_fail_on_warn_allows_warnings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    y = _write_yaml(
        tmp_path,
        textwrap.dedent(
            """
            version: "0.1"
            steps:
              - name: s1
                agent: { id: "flujo.builtins.stringify" }
                input: "hello"
                updates_context: true
              - name: s2
                agent: { id: "flujo.builtins.stringify" }
                input: "{{ previous_step.output }}"
            """
        ),
    )

    # Should not raise; writes JSON to stdout containing warnings
    _validate_impl(
        str(y),
        strict=False,
        output_format="json",
        include_imports=False,
        fail_on_warn=False,
        rules=None,
    )
    out = capsys.readouterr().out
    assert out.strip().startswith("{")
