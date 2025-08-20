from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_run_pipeline_in_memory_mocks_side_effects(tmp_path: Path) -> None:
    from tests.conftest import get_registered_factory

    # Ensure the fs_write_file skill is registered (side_effects=True in builtins)
    _ = get_registered_factory("flujo.builtins.fs_write_file")

    yaml_text = (
        'version: "0.1"\n'
        "steps:\n"
        "- name: WriteFile\n"
        "  agent:\n"
        "    id: flujo.builtins.fs_write_file\n"
        "    params:\n"
        "      path: should_not_exist.txt\n"
        "      content: hello\n"
    )

    factory = get_registered_factory("flujo.builtins.run_pipeline_in_memory")
    run_mem = factory()

    out = await run_mem(yaml_text=yaml_text, input_text="", sandbox=True, base_dir=str(tmp_path))
    assert isinstance(out, dict)
    result = out.get("dry_run_result")
    # Walk the step history to find mocked output
    mocked_found = False
    try:
        for sr in getattr(result, "step_history", []) or []:
            if isinstance(getattr(sr, "output", None), dict) and sr.output.get("mocked"):
                mocked_found = True
                break
    except Exception:
        pass
    assert mocked_found, "Expected side-effect skill to be mocked in dry run"
    # Ensure no file was created
    assert not (tmp_path / "should_not_exist.txt").exists()
