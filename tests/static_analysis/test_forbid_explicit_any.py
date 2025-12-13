from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.fast
def test_type_safety_lint_blocks_explicit_any() -> None:
    """Run lint_type_safety to ensure explicit Any is forbidden in core/DSL."""
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "lint_type_safety.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr + res.stdout
