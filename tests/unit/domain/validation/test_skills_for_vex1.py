"""Test helper skills for V-EX1 validation tests."""

from typing import Any, Optional
from flujo.domain.models import PipelineContext


async def custom_skill_example(data: Any, *, context: Optional[PipelineContext] = None) -> str:
    """Example custom skill for testing V-EX1 linter."""
    return f"Processed: {data}"


def custom_sync_skill(data: Any, *, context: Optional[PipelineContext] = None) -> str:
    """Another example custom skill."""
    return f"Sync processed: {data}"
