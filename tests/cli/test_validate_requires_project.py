from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pytest


def test_validate_errors_without_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "validate", "--format=json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    out = (result.stdout or "") + (result.stderr or "")
    assert "Not a Flujo project" in out
