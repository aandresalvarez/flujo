"""Security tests for state deserialization."""

import json
from pathlib import Path

import pytest

from flujo.state.backends.file import FileBackend


@pytest.mark.asyncio
async def test_resuming_from_malicious_state_fails_safely(tmp_path: Path) -> None:
    backend = FileBackend(tmp_path)
    run_id = "malicious"
    malicious = {
        "__type__": "builtins.eval",
        "payload": "__import__('pathlib').Path('pwned').write_text('x')",
    }
    state_file = tmp_path / f"{run_id}.json"
    state_file.write_text(json.dumps(malicious))

    assert not (tmp_path / "pwned").exists()
    result = await backend.load_state(run_id)
    assert result == malicious
    assert not (tmp_path / "pwned").exists()
