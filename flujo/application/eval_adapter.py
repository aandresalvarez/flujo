"""Utilities for integrating :class:`Flujo` with pydantic-evals."""

from typing import Any

from .runner import Flujo
from ..domain.models import PipelineResult


async def run_pipeline_async(inputs: Any, *, runner: Flujo[Any, Any, Any]) -> PipelineResult[Any]:
    """Adapter to run a :class:`Flujo` engine as a pydantic-evals task."""
    result: PipelineResult[Any] | None = None
    async for item in runner.run_async(inputs):
        result = item
    assert result is not None
    return result


# Example usage:
# runner: Flujo[Any, Any] = Flujo(your_pipeline_or_step)
