"""Typed test fixtures for Flujo.

This module provides type-safe factory functions for creating test data.
These fixtures replace ad-hoc test object creation with typed, reusable factories.

Usage:
    from tests.test_types.fixtures import create_test_step, create_test_step_result

    def test_my_feature():
        step = create_test_step(name="test")
        result = create_test_step_result(name="test", output="data")
"""

from typing import Any, Optional
from flujo.domain.dsl.step import Step, StepConfig
from flujo.domain.models import StepResult, UsageLimits
from flujo.domain.dsl.pipeline import Pipeline


def create_test_step(
    name: str = "test_step",
    agent: Optional[Any] = None,
    config: Optional[StepConfig] = None,
    **kwargs: Any,
) -> Step[Any, Any]:
    """Create a test Step with type-safe defaults.

    Args:
        name: Step name
        agent: Optional agent for the step
        config: Optional step configuration
        **kwargs: Additional Step fields

    Returns:
        A Step instance with defaults applied
    """
    return Step(name=name, agent=agent, config=config or StepConfig(), **kwargs)


def create_test_step_result(
    name: str = "test_step",
    output: Any = None,
    success: bool = True,
    feedback: Optional[str] = None,
    **kwargs: Any,
) -> StepResult:
    """Create a test StepResult with type-safe defaults.

    Args:
        name: Step name
        output: Step output value
        success: Whether the step succeeded
        feedback: Optional feedback message
        **kwargs: Additional StepResult fields

    Returns:
        A StepResult instance with defaults applied
    """
    return StepResult(name=name, output=output, success=success, feedback=feedback, **kwargs)


def create_test_pipeline(
    steps: Optional[list[Step[Any, Any]]] = None, **kwargs: Any
) -> Pipeline[Any, Any]:
    """Create a test Pipeline with type-safe defaults.

    Args:
        steps: List of steps for the pipeline
        **kwargs: Additional Pipeline fields

    Returns:
        A Pipeline instance with defaults applied
    """
    return Pipeline(steps=steps or [], **kwargs)


def create_test_usage_limits(
    total_cost_usd_limit: float = 100.0, total_tokens_limit: int = 10000, **kwargs: Any
) -> UsageLimits:
    """Create test UsageLimits with type-safe defaults.

    Args:
        total_cost_usd_limit: Maximum cost in USD
        total_tokens_limit: Maximum tokens
        **kwargs: Additional UsageLimits fields

    Returns:
        A UsageLimits instance with defaults applied
    """
    return UsageLimits(
        total_cost_usd_limit=total_cost_usd_limit, total_tokens_limit=total_tokens_limit, **kwargs
    )


# Predefined test fixtures for common scenarios
TEST_STEP_RESULT_SUCCESS = create_test_step_result(
    name="test_step", output="test_output", success=True
)

TEST_STEP_RESULT_FAILURE = create_test_step_result(
    name="test_step", output=None, success=False, feedback="Test failure"
)

__all__ = [
    "create_test_step",
    "create_test_step_result",
    "create_test_pipeline",
    "create_test_usage_limits",
    "TEST_STEP_RESULT_SUCCESS",
    "TEST_STEP_RESULT_FAILURE",
]
