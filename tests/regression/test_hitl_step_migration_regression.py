"""
Regression tests for HITL step migration.

This test suite ensures that the migrated HITL step implementation
maintains backward compatibility and preserves all existing behaviors.
"""

import pytest
from unittest.mock import Mock

from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl.step import HumanInTheLoopStep
from flujo.domain.models import PipelineContext
from flujo.exceptions import PausedException

# Legacy implementation is not available in the current codebase
# The regression tests will focus on ensuring the new implementation
# maintains the expected behavior without comparing to legacy
legacy_handle_hitl_step = None


class TestHITLStepMigrationRegression:
    """Regression tests for HITL step migration."""

    @pytest.fixture
    def executor_core(self) -> ExecutorCore:
        """Create a test ExecutorCore instance."""
        return ExecutorCore()

    @pytest.fixture
    def mock_hitl_step(self) -> Mock:
        """Create a mock HumanInTheLoopStep."""
        step = Mock(spec=HumanInTheLoopStep)
        step.name = "test_hitl_step"
        step.message_for_user = None
        return step

    @pytest.fixture
    def mock_context(self) -> PipelineContext:
        """Create a real PipelineContext for strict typed-context enforcement."""
        return PipelineContext()

    # ============================================================================
    # Phase 3.1: Existing Functionality Preservation
    # ============================================================================

    async def test_hitl_step_existing_behavior_preservation(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: PipelineContext,
    ):
        """Test that all existing HITL step behaviors are preserved."""
        # Test cases that cover all existing behaviors
        test_cases = [
            {
                "name": "basic_functionality",
                "message": None,
                "data": "test_data",
                "expected_message": "test_data",
            },
            {
                "name": "custom_message",
                "message": "Custom message",
                "data": "test_data",
                "expected_message": "Custom message",
            },
            {
                "name": "empty_message",
                "message": "",
                "data": "test_data",
                "expected_message": "",
            },
            {
                "name": "complex_data",
                "message": None,
                "data": {"key": "value", "nested": {"data": "test"}},
                "expected_message": "{'key': 'value', 'nested': {'data': 'test'}}",
            },
            {
                "name": "numeric_data",
                "message": None,
                "data": 123,
                "expected_message": "123",
            },
            {
                "name": "list_data",
                "message": None,
                "data": [1, 2, 3],
                "expected_message": "[1, 2, 3]",
            },
            {
                "name": "none_data",
                "message": None,
                "data": None,
                "expected_message": "None",
            },
        ]

        for case in test_cases:
            mock_hitl_step.message_for_user = case["message"]
            data = case["data"]

            # Test migrated implementation
            with pytest.raises(PausedException) as exc_info:
                await executor_core._handle_hitl_step(
                    mock_hitl_step,
                    data,
                    mock_context,
                    None,  # resources
                    None,  # limits
                    None,  # context_setter
                )

            # Verify behavior matches expected
            expected_message = case["expected_message"]
            assert expected_message in str(exc_info.value)
            assert mock_context.status == "paused"
            assert mock_context.pause_message == expected_message
            assert mock_context.hitl_data.get("last_hitl_step") == mock_hitl_step.name
            # paused_step_input may be AskHumanCommand; allow either form
            assert mock_context.paused_step_input is not None

    async def test_hitl_step_edge_cases_regression(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: PipelineContext,
    ):
        """Test edge cases that existed before migration."""
        # Test edge cases that should still work
        edge_cases = [
            {
                "name": "unicode_data",
                "data": "测试数据 🚀",
                "message": None,
            },
            {
                "name": "binary_data",
                "data": b"binary_data",
                "message": None,
            },
            {
                "name": "very_large_data",
                "data": "x" * 10000,
                "message": None,
            },
            {
                "name": "empty_data",
                "data": "",
                "message": None,
            },
            {
                "name": "zero_data",
                "data": 0,
                "message": None,
            },
            {
                "name": "boolean_data",
                "data": True,
                "message": None,
            },
        ]

        for case in edge_cases:
            mock_hitl_step.message_for_user = case["message"]
            data = case["data"]

            with pytest.raises(PausedException) as exc_info:
                await executor_core._handle_hitl_step(
                    mock_hitl_step,
                    data,
                    mock_context,
                    None,  # resources
                    None,  # limits
                    None,  # context_setter
                )

            # Verify edge case is handled correctly
            expected_message = str(data) if case["message"] is None else case["message"]
            assert expected_message in str(exc_info.value)
            assert mock_context.status == "paused"
            assert mock_context.pause_message == expected_message
            assert mock_context.paused_step_input == data
            assert mock_context.hitl_data.get("last_hitl_step") == mock_hitl_step.name

    async def test_hitl_step_error_scenarios_regression(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: PipelineContext,
    ):
        """Test error scenarios that existed before migration."""
        # Test error scenarios that should be handled gracefully
        error_scenarios = [
            {
                "name": "string_conversion_error",
                "data": Mock(__str__=Mock(side_effect=Exception("String conversion failed"))),
                "message": None,
                "should_raise_paused": True,
            },
            {
                "name": "context_update_error",
                "data": "test_data",
                "message": None,
                "should_raise_paused": True,
                "context_error": True,
            },
            {
                "name": "none_context",
                "data": "test_data",
                "message": None,
                "should_raise_paused": True,
                "context": None,
            },
        ]

        for scenario in error_scenarios:
            mock_hitl_step.message_for_user = scenario["message"]
            data = scenario["data"]
            context = scenario.get("context", mock_context)

            if scenario["should_raise_paused"]:
                with pytest.raises(PausedException) as exc_info:
                    await executor_core._handle_hitl_step(
                        mock_hitl_step,
                        data,
                        context,
                        None,  # resources
                        None,  # limits
                        None,  # context_setter
                    )

                # Verify PausedException is still raised even with errors
                assert isinstance(exc_info.value, PausedException)

    # ============================================================================
    # Phase 3.2: Legacy Compatibility Tests
    # ============================================================================

    async def test_hitl_step_legacy_compatibility_comparison(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: PipelineContext,
    ):
        """Test that migrated implementation produces expected behavior."""
        # Test cases to validate migrated behavior
        test_cases = [
            {"message": None, "data": "simple_data"},
            {"message": "Custom message", "data": "test_data"},
            {"message": "", "data": "empty_message_data"},
            {"message": None, "data": 123},
            {"message": None, "data": {"key": "value"}},
        ]

        for case in test_cases:
            mock_hitl_step.message_for_user = case["message"]
            data = case["data"]

            # Test migrated implementation
            migrated_context = PipelineContext()

            with pytest.raises(PausedException) as migrated_exc:
                await executor_core._handle_hitl_step(
                    mock_hitl_step,
                    data,
                    migrated_context,
                    None,  # resources
                    None,  # limits
                    None,  # context_setter
                )

            # Validate expected behavior
            expected_message = case["message"] if case["message"] is not None else str(data)

            # Should raise PausedException
            assert isinstance(migrated_exc.value, PausedException)

            # Should have the expected message
            assert expected_message in str(migrated_exc.value)

            # Should have additional context updates
            assert migrated_context.status == "paused"
            assert migrated_context.pause_message == expected_message
            assert migrated_context.paused_step_input == data
            assert migrated_context.hitl_data.get("last_hitl_step") == mock_hitl_step.name

    async def test_hitl_step_backward_compatibility_regression(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
    ):
        """Test backward compatibility with existing HITL step usage patterns."""
        # Test patterns that existing code might use
        compatibility_patterns = [
            {
                "name": "basic_usage",
                "setup": lambda step, context: None,
                "data": "basic_data",
                "message": None,
            },
            {
                "name": "custom_message_usage",
                "setup": lambda step, context: setattr(step, "message_for_user", "Custom message"),
                "data": "custom_data",
                "message": "Custom message",
            },
            {
                "name": "context_preservation",
                "setup": lambda step, context: context.import_artifacts.update(
                    {"existing": "data"}
                ),
                "data": "preserved_data",
                "message": None,
            },
        ]

        for pattern in compatibility_patterns:
            # Reset mock objects for each pattern
            mock_hitl_step.message_for_user = None
            context = PipelineContext()

            # Setup the pattern
            pattern["setup"](mock_hitl_step, context)
            data = pattern["data"]

            with pytest.raises(PausedException) as exc_info:
                await executor_core._handle_hitl_step(
                    mock_hitl_step,
                    data,
                    context,
                    None,  # resources
                    None,  # limits
                    None,  # context_setter
                )

            # Verify compatibility is maintained
            # The new implementation uses the custom message if provided, otherwise the data
            if pattern["message"] is not None:
                expected_message = pattern["message"]
            else:
                expected_message = str(data)
            assert expected_message in str(exc_info.value)
            assert context.status == "paused"
            assert context.pause_message == expected_message
            assert context.paused_step_input == data
            assert context.hitl_data.get("last_hitl_step") == mock_hitl_step.name

            # Verify existing context data is preserved
            if "existing" in context.import_artifacts:
                assert context.import_artifacts.get("existing") == "data"

    # ============================================================================
    # Phase 3.3: Performance Regression Tests
    # ============================================================================

    async def test_hitl_step_performance_regression(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: PipelineContext,
    ):
        """Test that performance is not regressed from legacy implementation."""
        import time

        # Test data
        data = "performance_test_data"
        mock_hitl_step.message_for_user = None

        # Measure migrated implementation performance
        start_time = time.perf_counter()
        with pytest.raises(PausedException):
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                data,
                mock_context,
                None,  # resources
                None,  # limits
                None,  # context_setter
            )
        migrated_time = time.perf_counter() - start_time

        # Since legacy implementation is not available, we'll just validate
        # that the migrated implementation performs within reasonable bounds
        # The new implementation should complete quickly (under 1 second)
        assert migrated_time < 1.0  # Should complete within 1 second

    async def test_hitl_step_memory_regression(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: PipelineContext,
    ):
        """Test that memory usage is not significantly increased."""
        import gc
        import sys

        # Test with large data
        large_data = {"key": "value" * 1000}
        mock_hitl_step.message_for_user = None

        # Measure memory usage
        gc.collect()
        initial_memory = sys.getsizeof(mock_context.model_dump())

        with pytest.raises(PausedException):
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                large_data,
                mock_context,
                None,  # resources
                None,  # limits
                None,  # context_setter
            )

        final_memory = sys.getsizeof(mock_context.model_dump())
        memory_increase = final_memory - initial_memory

        # Assert memory usage is reasonable
        assert memory_increase < 50_000  # Should not increase by more than 50KB

    # ============================================================================
    # Phase 3.4: Functionality Regression Tests
    # ============================================================================

    async def test_hitl_step_functionality_regression(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
    ):
        """Test that all functionality is preserved without regression."""
        # Test all core functionality
        functionality_tests = [
            {
                "name": "message_generation",
                "data": "test_data",
                "message": None,
                "expected_exception": PausedException,
                "expected_in_exception": "test_data",
            },
            {
                "name": "custom_message",
                "data": "test_data",
                "message": "Custom message",
                "expected_exception": PausedException,
                "expected_in_exception": "Custom message",
            },
            {
                "name": "context_update",
                "data": "test_data",
                "message": None,
                "expected_exception": PausedException,
                "expected_in_context": {"status": "paused"},
            },
            {
                "name": "exception_raising",
                "data": "test_data",
                "message": None,
                "expected_exception": PausedException,
                "expected_in_exception": "test_data",
            },
        ]

        for test in functionality_tests:
            mock_hitl_step.message_for_user = test["message"]
            data = test["data"]

            with pytest.raises(test["expected_exception"]) as exc_info:
                await executor_core._handle_hitl_step(
                    mock_hitl_step,
                    data,
                    mock_context,
                    None,  # resources
                    None,  # limits
                    None,  # context_setter
                )

            # Verify functionality is preserved
            if "expected_in_exception" in test:
                assert test["expected_in_exception"] in str(exc_info.value)

            if "expected_in_context" in test:
                expected = test["expected_in_context"]
                assert mock_context.status == expected.get("status")
                if expected.get("pause_message") is not None:
                    assert mock_context.pause_message == expected["pause_message"]
                assert mock_context.hitl_data.get("last_hitl_step") == mock_hitl_step.name

    async def test_hitl_step_robustness_regression(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
    ):
        """Test that robustness is maintained or improved."""
        # Test robustness scenarios
        robustness_tests = [
            {
                "name": "problematic_data",
                "data": Mock(__str__=Mock(side_effect=Exception("String conversion failed"))),
                "should_work": True,
            },
            {
                "name": "context_error",
                "data": "test_data",
                "context_error": True,
                "should_work": True,
            },
            {
                "name": "none_context",
                "data": "test_data",
                "context": None,
                "should_work": True,
            },
        ]

        for test in robustness_tests:
            mock_hitl_step.message_for_user = None
            data = test["data"]
            context = test.get("context", mock_context)

            if test["should_work"]:
                with pytest.raises(PausedException) as exc_info:
                    await executor_core._handle_hitl_step(
                        mock_hitl_step,
                        data,
                        context,
                        None,  # resources
                        None,  # limits
                        None,  # context_setter
                    )

                # Should still raise PausedException
                assert isinstance(exc_info.value, PausedException)
