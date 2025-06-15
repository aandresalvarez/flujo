"""Utilities for integrating PipelineRunner with pydantic-evals."""

from typing import Any

from .pipeline_runner import PipelineRunner
from ..domain.models import PipelineResult


async def run_pipeline_async(inputs: Any, *, runner: PipelineRunner[Any, Any]) -> PipelineResult:
    """Adapter to run a PipelineRunner as a pydantic-evals task."""
    return await runner.run_async(inputs)

# Example usage:
# runner: PipelineRunner[Any, Any] = PipelineRunner(your_pipeline_or_step)
