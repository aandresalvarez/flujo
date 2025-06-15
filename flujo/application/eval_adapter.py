"""Utilities for integrating :class:`Flujo` with pydantic-evals."""

from typing import Any

from .flujo_engine import Flujo
from ..domain.models import PipelineResult


async def run_pipeline_async(inputs: Any, *, runner: Flujo[Any, Any]) -> PipelineResult:
    """Adapter to run a :class:`Flujo` engine as a pydantic-evals task."""
    return await runner.run_async(inputs)


# Example usage:
# runner: Flujo[Any, Any] = Flujo(your_pipeline_or_step)
