"""Type-safe test utilities for Flujo.

This module provides typed fixtures, mocks, and test utilities to improve
type safety in Flujo's test suite.
"""

# Import fixtures and mocks for easy access
# These will be implemented incrementally as we migrate tests

from .fakes import FakeAgent, FakeUsageMeter  # noqa: F401
from .fixtures import (  # noqa: F401
    TEST_STEP_RESULT_FAILURE,
    TEST_STEP_RESULT_SUCCESS,
    create_test_pipeline,
    create_test_step,
    create_test_step_result,
    create_test_usage_limits,
)
from .mocks import create_mock_executor_core  # noqa: F401

__all__ = [
    "FakeAgent",
    "FakeUsageMeter",
    "TEST_STEP_RESULT_FAILURE",
    "TEST_STEP_RESULT_SUCCESS",
    "create_mock_executor_core",
    "create_test_pipeline",
    "create_test_step",
    "create_test_step_result",
    "create_test_usage_limits",
]
