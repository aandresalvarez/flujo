from pathlib import Path

import pytest

from flujo.domain import Step
from flujo.domain.models import PipelineContext
from flujo.state.backends.sqlite import SQLiteBackend


@pytest.mark.asyncio
async def test_runner_uses_sqlite_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    # Temporarily disable test mode to test actual default behavior
    monkeypatch.delenv("FLUJO_TEST_MODE", raising=False)
    # Also ensure CI environment variables don't interfere
    monkeypatch.delenv("CI", raising=False)
    # Clear any explicit state backend override from test fixtures
    monkeypatch.delenv("FLUJO_STATE_URI", raising=False)
    monkeypatch.delenv("FLUJO_CONFIG_PATH", raising=False)

    async def s(data: int) -> int:
        return data + 1

    pipeline = Step.from_callable(s, name="s")
    # Don't use create_test_flujo here since we want to test the actual default behavior
    from flujo.application.runner import Flujo

    runner = Flujo(pipeline, context_model=PipelineContext)
    assert isinstance(runner.state_backend, SQLiteBackend)
    assert runner.state_backend.db_path == tmp_path / "flujo_ops.db"
