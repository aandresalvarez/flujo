from pathlib import Path

import pytest

from flujo.domain import Step
from flujo.domain.models import PipelineContext
from flujo.state.backends.sqlite import SQLiteBackend
from tests.conftest import create_test_flujo


@pytest.mark.asyncio
async def test_runner_uses_sqlite_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    async def s(data: int) -> int:
        return data + 1

    pipeline = Step.from_callable(s, name="s")
    runner = create_test_flujo(pipeline, context_model=PipelineContext)
    assert isinstance(runner.state_backend, SQLiteBackend)
    assert runner.state_backend.db_path == tmp_path / "flujo_ops.db"
