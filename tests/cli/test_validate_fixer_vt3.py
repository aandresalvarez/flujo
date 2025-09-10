from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_validate_fix_vt3_corrects_common_filter_typos(tmp_path: Path) -> None:
    yml = (
        'version: "0.1"\n'
        "steps:\n"
        '  - name: A\n    agent: { id: "flujo.builtins.stringify" }\n    input: "hello"\n'
        '  - name: B\n    agent: { id: "flujo.builtins.stringify" }\n    input: "{{ previous_step | lowercase }} {{ previous_step | to_json }}"\n'
    )
    f = tmp_path / "p.yaml"
    f.write_text(yml)

    # First run should show V-T3 warnings
    res0 = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "validate", str(f), "--format=json"],
        capture_output=True,
        text=True,
    )
    assert '"V-T3"' in (res0.stdout or "{}")

    # Apply fixer
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "validate",
            str(f),
            "--fix",
            "--yes",
            "--format=json",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode in (0, 4)
    fixed = f.read_text()
    assert "| lower" in fixed
    assert "| tojson" in fixed

    # After fix, re-run validate should not report V-T3
    res2 = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "validate", str(f), "--format=json"],
        capture_output=True,
        text=True,
    )
    out = res2.stdout or "{}"
    assert '"V-T3"' not in out
