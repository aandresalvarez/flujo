import pytest

from pathlib import Path

from flujo.application.core.state_manager import StateManager
from flujo.domain.models import PipelineContext, ConversationTurn, ConversationRole
from flujo.state.backends.sqlite import SQLiteBackend


@pytest.mark.slow
@pytest.mark.serial
async def test_conversation_history_persists_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    backend = SQLiteBackend(db_path)
    sm: StateManager[PipelineContext] = StateManager(state_backend=backend)

    run_id = "conv-run-1"
    ctx = PipelineContext(initial_prompt="hello")
    ctx.conversation_history.append(
        ConversationTurn(role=ConversationRole.user, content="Hi there")
    )

    await sm.persist_workflow_state(
        run_id=run_id,
        context=ctx,
        current_step_index=0,
        last_step_output=None,
        status="running",
        state_created_at=None,
        step_history=[],
    )

    loaded_ctx, last_output, idx, created_at, name, ver, sh = await sm.load_workflow_state(
        run_id, PipelineContext
    )

    assert loaded_ctx is not None
    assert isinstance(loaded_ctx, PipelineContext)
    assert loaded_ctx.initial_prompt == "hello"
    assert len(loaded_ctx.conversation_history) == 1
    turn = loaded_ctx.conversation_history[0]
    assert turn.role == ConversationRole.user
    assert turn.content == "Hi there"
