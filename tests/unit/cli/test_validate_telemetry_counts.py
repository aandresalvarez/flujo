from __future__ import annotations
import json
import textwrap
from pathlib import Path

from flujo.cli.main import _validate_impl


def test_json_counts_when_telemetry_enabled(tmp_path: Path, monkeypatch) -> None:
    yaml_text = textwrap.dedent(
        """
        version: "0.1"
        steps:
          - name: A
            agent: { id: "flujo.builtins.stringify" }
            input: "hello"
            updates_context: true
          - name: B
            agent: { id: "flujo.builtins.stringify" }
            input: "{{ previous_step.output }}"
        """
    )
    p = tmp_path / "p.yaml"
    p.write_text(yaml_text)
    monkeypatch.setenv("FLUJO_CLI_TELEMETRY", "1")
    _validate_impl(
        str(p),
        strict=False,
        output_format="json",
        include_imports=False,
        fail_on_warn=False,
        rules=None,
        explain=False,
    )
    # Capture stdout into a buffer for JSON parsing
    out_file = tmp_path / "out.json"
    # Run again, redirecting stdout to a file manually
    import io

    buf = io.StringIO()
    import contextlib

    with contextlib.redirect_stdout(buf):
        _validate_impl(
            str(p),
            strict=False,
            output_format="json",
            include_imports=False,
            fail_on_warn=False,
            rules=None,
            explain=False,
        )
    out_file.write_text(buf.getvalue())
    data = json.loads(out_file.read_text())
    counts = data.get("counts") or {}
    # Expect at least one warning (V-T1) counted
    assert "warning" in counts and any(counts["warning"].values())
