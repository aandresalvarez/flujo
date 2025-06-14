"""Utilities for integrating PipelineRunner with pydantic-evals."""

from typing import Any

from .pipeline_runner import PipelineRunner
from ..domain.models import PipelineResult


async def run_pipeline_async(inputs: Any, *, runner: PipelineRunner) -> PipelineResult:
    """Adapter to run a PipelineRunner as a pydantic-evals task."""
    return await runner.run_async(inputs)
