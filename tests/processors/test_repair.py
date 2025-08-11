import pytest

from flujo.agents.repair import DeterministicRepairProcessor, MAX_LITERAL_EVAL_SIZE


@pytest.mark.asyncio
async def test_literal_eval_size_guard() -> None:
    proc = DeterministicRepairProcessor()
    oversized = "x" * (MAX_LITERAL_EVAL_SIZE + 1)
    with pytest.raises(ValueError, match="Input too large for safe literal evaluation."):
        await proc.process(oversized)
