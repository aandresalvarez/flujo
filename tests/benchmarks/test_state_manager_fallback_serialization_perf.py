"""Benchmark tests for StateManager fallback serialization."""

from __future__ import annotations

import os
from datetime import datetime

import pytest
from pydantic import Field

from flujo.application.core.state_manager import StateManager
from flujo.domain.models import PipelineContext
from flujo.type_definitions.common import JSONObject


pytestmark = [pytest.mark.benchmark, pytest.mark.slow]


class ExtendedPipelineContext(PipelineContext):
    """Concrete context with additional metadata for fallback serialization."""

    pipeline_id: str = "perf_test_123"
    pipeline_name: str = "Performance Test"
    pipeline_version: str = "1.0.0"
    total_steps: int = 10
    status: str | None = None
    current_step: int = 0
    last_error: str | None = None
    metadata: JSONObject = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@pytest.fixture()
def state_manager() -> StateManager[PipelineContext]:
    return StateManager()


@pytest.fixture()
def pipeline_context() -> ExtendedPipelineContext:
    context = ExtendedPipelineContext(
        run_id="perf_run_456",
        initial_prompt="performance test",
    )
    context.pipeline_id = "perf_test_123"
    context.pipeline_name = "Performance Test"
    context.pipeline_version = "1.0.0"
    context.total_steps = 10
    context.created_at = datetime.now()
    context.updated_at = datetime.now()
    context.status = "running"
    context.current_step = 2
    context.last_error = "previous error"
    context.metadata = {"key": "value"}
    return context


def test_fallback_serialization_benchmark(benchmark, state_manager, pipeline_context):
    """Ensure fallback serialization stays within acceptable latency limits."""

    def exercise() -> None:
        state_manager._build_context_fallback(
            pipeline_context, error_message="Failed to serialize context"
        )

    benchmark(exercise)

    threshold = float(os.getenv("FLUJO_FALLBACK_BENCH_THRESHOLD", "0.10"))
    assert benchmark.stats["min"] < threshold, (
        f"Fallback serialization regression: {benchmark.stats['min']:.6f}s >= {threshold:.3f}s"
    )
