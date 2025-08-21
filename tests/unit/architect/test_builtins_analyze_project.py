from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_analyze_project_detects_common_files(tmp_path: Path) -> None:
    # Arrange: create a simple project layout
    (tmp_path / "pyproject.toml").write_text("[build-system]\nrequires=[]\n")
    (tmp_path / "requirements.txt").write_text("flujo\n")

    # Acquire factory via registry helper (ensures builtins are registered)
    from tests.conftest import get_registered_factory

    factory = get_registered_factory("flujo.builtins.analyze_project")
    analyze = factory(directory=str(tmp_path))

    # Act
    out = await analyze(None)

    # Assert
    assert isinstance(out, dict)
    summary = out.get("project_summary", "")
    assert "Found" in summary
    assert "pyproject.toml" in summary
    assert "requirements.txt" in summary


@pytest.mark.asyncio
async def test_analyze_project_empty_dir(tmp_path: Path) -> None:
    from tests.conftest import get_registered_factory

    factory = get_registered_factory("flujo.builtins.analyze_project")
    analyze = factory(directory=str(tmp_path))

    out = await analyze(None)
    assert isinstance(out, dict)
    summary = out.get("project_summary", "")
    assert "Found" in summary
