from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_validate_fix_vc2_changes_parent_scratchpad_to_key(tmp_path: Path) -> None:
    # Build a minimal parent with an import step output mapping to scratchpad root
    child = (
        'version: "0.1"\n'
        "steps:\n"
        '  - name: C\n    agent: { id: "flujo.builtins.stringify" }\n    input: "hi"\n'
    )
    (tmp_path / "child.yaml").write_text(child)
    parent = (
        'version: "0.1"\n'
        'imports:\n  c: "child.yaml"\n'
        "steps:\n"
        '  - name: RunChild\n    uses: imports.c\n    updates_context: true\n    config:\n      outputs:\n        - { child: "scratchpad.value", parent: "scratchpad" }\n'
    )
    f = tmp_path / "p.yaml"
    f.write_text(parent)

    # First run should show V-C2 warning
    res0 = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "validate", str(f), "--format=json"],
        capture_output=True,
        text=True,
    )
    assert '"V-C2"' in (res0.stdout or "{}")

    # Apply fixer only for V-C2 via env filter
    env = dict(**os_environ_copy())
    env["FLUJO_FIX_RULES"] = "V-C2"
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
        env=env,
    )
    assert res.returncode in (0, 4)
    fixed = f.read_text()
    assert "parent: scratchpad.value" in fixed
    assert 'parent: "scratchpad"' not in fixed and "parent: scratchpad\n" not in fixed

    # After fix, re-run validate should not report V-C2
    res2 = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "validate", str(f), "--format=json"],
        capture_output=True,
        text=True,
    )
    out = res2.stdout or "{}"
    assert '"V-C2"' not in out


def os_environ_copy() -> dict:
    import os

    return dict(os.environ)
