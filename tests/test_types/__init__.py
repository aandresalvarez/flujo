"""Type-safe test utilities for Flujo.

This module provides typed fixtures, mocks, and test utilities to improve
type safety in Flujo's test suite.
"""

# Import fixtures and mocks for easy access
# These will be implemented incrementally as we migrate tests

from .fakes import FakeAgent, FakeUsageMeter, FakeCacheBackend, TestContext
from .fixtures import (
    TEST_STEP_RESULT_FAILURE,
    TEST_STEP_RESULT_SUCCESS,
    create_test_context,
    create_test_pipeline,
    create_test_step,
    create_test_step_result,
    create_test_usage_limits,
)
from .mocks import create_mock_executor_core

__all__ = [
    "FakeAgent",
    "FakeUsageMeter",
    "FakeCacheBackend",
    "TestContext",
    "TEST_STEP_RESULT_FAILURE",
    "TEST_STEP_RESULT_SUCCESS",
    "create_mock_executor_core",
    "create_test_context",
    "create_test_pipeline",
    "create_test_step",
    "create_test_step_result",
    "create_test_usage_limits",
]
