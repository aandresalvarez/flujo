"""Recipe modules for common workflow patterns.

This module provides factory functions for common workflow patterns.

RECOMMENDED: Use the factory functions for better transparency, composability, and future YAML/AI support:
- make_default_pipeline() - Creates a Review → Solution → Validate pipeline
- make_agentic_loop_pipeline() - Creates an explorative agent workflow
- run_default_pipeline() - Executes a default pipeline
- run_agentic_loop_pipeline() - Executes an agentic loop pipeline
"""

# Factory functions (recommended)
from .factories import (
    make_default_pipeline,
    make_state_machine_pipeline,
    make_agentic_loop_pipeline,
    run_default_pipeline,
    run_agentic_loop_pipeline,
)

# Legacy classes (deprecated) - removed in FSD-007

__all__ = [
    # Factory functions (recommended)
    "make_default_pipeline",
    "make_state_machine_pipeline",
    "make_agentic_loop_pipeline",
    "run_default_pipeline",
    "run_agentic_loop_pipeline",
]
