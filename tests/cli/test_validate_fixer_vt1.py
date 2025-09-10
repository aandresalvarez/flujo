from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_validate_fix_vt1_rewrites_yaml(tmp_path: Path) -> None:
    yml = (
        'version: "0.1"\n'
        "steps:\n"
        '  - name: A\n    agent: { id: "flujo.builtins.stringify" }\n    input: "hello"\n'
        '  - name: B\n    agent: { id: "flujo.builtins.stringify" }\n    input: "{{ previous_step.output }}"\n'
    )
    f = tmp_path / "p.yaml"
    f.write_text(yml)

    # Run fixer
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
    assert "previous_step.output" not in fixed
    assert "previous_step | tojson" in fixed
    # After fix, re-run validate should not report V-T1
    res2 = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "validate", str(f), "--format=json"],
        capture_output=True,
        text=True,
    )
    out = res2.stdout or "{}"
    assert '"V-T1"' not in out
