"""
Flujo - A modern, type-safe framework for building AI-powered applications.

Flujo provides a robust foundation for creating AI agents, pipelines, and workflows
with structured outputs, comprehensive error handling, and excellent developer experience.

Key Features:
- Type-safe AI agent orchestration
- Structured output validation
- Comprehensive error handling and retry logic
- Async-first design with high performance
- Built-in caching and state management
- Extensive testing and debugging tools
- Production-ready telemetry and observability

Quick Start:
    from flujo import Step, Pipeline
    from flujo.agents import make_agent_async

    # Create an agent
    agent = make_agent_async("openai:gpt-4o", "You are a helpful assistant.", str)

    # Create a pipeline
    pipeline = Step.solution(agent)

    # Run the pipeline
    result = await pipeline.run("Hello, world!")
    print(result.output)
"""

# Performance optimizations are handled by flujo.utils.performance module

from .application.runner import Flujo
from .domain.dsl import Pipeline, Step
from .domain.dsl.step import StepConfig
from .domain.dsl.step import step  # Add back the step decorator
from .domain.models import PipelineResult
from .infra import init_telemetry
from .agents import make_agent_async  # Add back the agent factory
from .client import TaskClient
from .recipes.factories import (
    make_agentic_loop_pipeline,
    make_default_pipeline,
    run_agentic_loop_pipeline,
    run_default_pipeline,
)

# Ensure framework primitives are registered at import time
from . import framework as _framework  # noqa: F401

__version__ = "0.6.4"  # Security: fix CVEs in mcp, starlette, remove pytest-forked

__all__ = [
    "Flujo",
    "Pipeline",
    "Step",
    "StepConfig",
    "step",  # Add step decorator
    "PipelineResult",
    "make_agent_async",  # Add agent factory
    "init_telemetry",
    "make_agentic_loop_pipeline",
    "make_default_pipeline",
    "run_agentic_loop_pipeline",
    "run_default_pipeline",
    "TaskClient",
]
