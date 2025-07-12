"""Recipe modules for common workflow patterns.

This module provides both new factory functions (recommended) and legacy class-based
recipes (deprecated) for common workflow patterns.

RECOMMENDED: Use the factory functions for better transparency, composability, and future YAML/AI support:
- make_default_pipeline() - Creates a Review → Solution → Validate pipeline
- make_agentic_loop_pipeline() - Creates an explorative agent workflow
- run_default_pipeline() - Executes a default pipeline
- run_agentic_loop_pipeline() - Executes an agentic loop pipeline

DEPRECATED: The class-based recipes are deprecated:
- Default - Use make_default_pipeline() instead
- AgenticLoop - Use make_agentic_loop_pipeline() instead
"""

# New factory functions (recommended)
from .factories import (
    make_default_pipeline,
    make_state_machine_pipeline,
    make_agentic_loop_pipeline,
    run_default_pipeline,
    run_agentic_loop_pipeline,
)

# Legacy classes (deprecated)
from .default import Default
from .agentic_loop import AgenticLoop

__all__ = [
    # Factory functions (recommended)
    "make_default_pipeline",
    "make_state_machine_pipeline",
    "make_agentic_loop_pipeline",
    "run_default_pipeline",
    "run_agentic_loop_pipeline",
    # Legacy classes (deprecated)
    "Default",
    "AgenticLoop",
]
