"""
Test suite for HumanInTheLoopStep migration to ExecutorCore.

This test suite validates the migration of HITL step logic from step_logic.py
to ExecutorCore._handle_hitl_step method, ensuring complete functionality
preservation and performance improvements.

Author: Flujo Team
Version: 1.0
"""

import pytest
from unittest.mock import Mock

from flujo.application.core.ultra_executor import ExecutorCore
from flujo.domain.dsl.step import HumanInTheLoopStep
from flujo.domain.models import PipelineContext, UsageLimits
from flujo.exceptions import PausedException


class TestExecutorCoreHITLStepMigration:
    """Test suite for HITL step migration to ExecutorCore."""

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
    def mock_context(self) -> Mock:
        """Create a mock PipelineContext."""
        context = Mock(spec=PipelineContext)
        context.scratchpad = {}
        return context

    @pytest.fixture
    def mock_resources(self) -> Mock:
        """Create mock resources."""
        return Mock()

    @pytest.fixture
    def mock_limits(self) -> Mock:
        """Create mock usage limits."""
        return Mock(spec=UsageLimits)

    @pytest.fixture
    def mock_context_setter(self) -> Mock:
        """Create a mock context setter function."""
        return Mock()

    # ============================================================================
    # Phase 1.1: Basic Functionality Tests
    # ============================================================================

    async def test_handle_hitl_step_basic_message(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test basic HITL step with default message."""
        # Arrange
        data = "test_data"
        mock_hitl_step.message_for_user = None

        # Act & Assert
        with pytest.raises(PausedException) as exc_info:
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )

        # Verify the exception message is the string representation of data
        assert str(data) in str(exc_info.value)

    async def test_handle_hitl_step_custom_message(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test HITL step with custom message for user."""
        # Arrange
        custom_message = "Please review this data"
        mock_hitl_step.message_for_user = custom_message
        data = "test_data"

        # Act & Assert
        with pytest.raises(PausedException) as exc_info:
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )

        # Verify the custom message is used
        assert custom_message in str(exc_info.value)

    async def test_handle_hitl_step_context_scratchpad(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test that context scratchpad is properly updated."""
        # Arrange
        data = "test_data"
        custom_message = "Review required"
        mock_hitl_step.message_for_user = custom_message

        # Act & Assert
        with pytest.raises(PausedException):
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )

        # Verify scratchpad was updated
        assert mock_context.scratchpad["status"] == "paused"
        assert mock_context.scratchpad["hitl_message"] == custom_message
        assert mock_context.scratchpad["hitl_data"] == data

    async def test_handle_hitl_step_paused_exception(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test that PausedException is raised correctly."""
        # Arrange
        data = "test_data"

        # Act & Assert
        with pytest.raises(PausedException) as exc_info:
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )

        # Verify PausedException is raised with correct message
        assert isinstance(exc_info.value, PausedException)
        assert str(data) in str(exc_info.value)

    async def test_handle_hitl_step_data_handling(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test handling of various data types in HITL step."""
        # Test with different data types
        test_cases = [
            ("string_data", "string_data"),
            (123, "123"),
            ({"key": "value"}, "{'key': 'value'}"),
            ([1, 2, 3], "[1, 2, 3]"),
            (None, "None"),
        ]

        for data, expected_message in test_cases:
            mock_hitl_step.message_for_user = None

            with pytest.raises(PausedException) as exc_info:
                await executor_core._handle_hitl_step(
                    mock_hitl_step,
                    data,
                    mock_context,
                    mock_resources,
                    mock_limits,
                    mock_context_setter,
                )

            assert expected_message in str(exc_info.value)

    # ============================================================================
    # Phase 1.2: Context Management Tests
    # ============================================================================

    async def test_handle_hitl_step_context_preservation(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test that context state is preserved during pause."""
        # Arrange
        original_scratchpad = {"existing_key": "existing_value"}
        mock_context.scratchpad = original_scratchpad.copy()
        data = "test_data"

        # Act & Assert
        with pytest.raises(PausedException):
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )

        # Verify original scratchpad data is preserved
        assert mock_context.scratchpad["existing_key"] == "existing_value"
        # Verify new HITL data is added
        assert mock_context.scratchpad["status"] == "paused"
        assert mock_context.scratchpad["hitl_data"] == data

    async def test_handle_hitl_step_scratchpad_updates(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test scratchpad updates in context."""
        # Arrange
        data = "test_data"
        custom_message = "Custom message"
        mock_hitl_step.message_for_user = custom_message

        # Act & Assert
        with pytest.raises(PausedException):
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )

        # Verify all expected scratchpad updates
        expected_updates = {
            "status": "paused",
            "hitl_message": custom_message,
            "hitl_data": data,
        }

        for key, value in expected_updates.items():
            assert mock_context.scratchpad[key] == value

    async def test_handle_hitl_step_context_isolation(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test context isolation during HITL step execution."""
        # Arrange
        data = "test_data"
        original_scratchpad = {"key1": "value1", "key2": "value2"}
        mock_context.scratchpad = original_scratchpad.copy()

        # Act & Assert
        with pytest.raises(PausedException):
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )

        # Verify original data is preserved and new data is added
        for key, value in original_scratchpad.items():
            assert mock_context.scratchpad[key] == value

        # Verify HITL-specific data is added
        assert "status" in mock_context.scratchpad
        assert "hitl_message" in mock_context.scratchpad
        assert "hitl_data" in mock_context.scratchpad

    async def test_handle_hitl_step_context_serialization(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test context serialization for pause/resume."""
        # Arrange
        complex_data = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "string": "test",
        }
        mock_hitl_step.message_for_user = "Complex data test"

        # Act & Assert
        with pytest.raises(PausedException):
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                complex_data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )

        # Verify complex data is stored in scratchpad
        assert mock_context.scratchpad["hitl_data"] == complex_data

    # ============================================================================
    # Phase 1.3: Error Handling Tests
    # ============================================================================

    async def test_handle_hitl_step_message_generation_errors(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test error handling in message generation."""
        # Arrange
        # Create data that might cause issues in string conversion
        problematic_data = Mock()
        problematic_data.__str__ = Mock(side_effect=Exception("String conversion failed"))

        # Act & Assert
        with pytest.raises(PausedException) as exc_info:
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                problematic_data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )

        # Should still raise PausedException even with problematic data
        assert isinstance(exc_info.value, PausedException)

    async def test_handle_hitl_step_context_update_errors(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test error handling during context updates."""
        # Arrange
        # Make scratchpad assignment raise an exception
        mock_context.scratchpad = Mock()
        mock_context.scratchpad.__setitem__ = Mock(
            side_effect=Exception("Scratchpad update failed")
        )
        data = "test_data"

        # Act & Assert
        with pytest.raises(PausedException) as exc_info:
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )

        # Should still raise PausedException even if context update fails
        assert isinstance(exc_info.value, PausedException)

    async def test_handle_hitl_step_exception_propagation(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test that exceptions are properly propagated."""
        # Arrange
        data = "test_data"

        # Act & Assert
        with pytest.raises(PausedException) as exc_info:
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )

        # Verify PausedException is raised with correct message
        assert isinstance(exc_info.value, PausedException)
        assert str(data) in str(exc_info.value)

    async def test_handle_hitl_step_invalid_context(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test handling of invalid context objects."""
        # Arrange
        invalid_context = "not_a_pipeline_context"
        data = "test_data"

        # Act & Assert
        with pytest.raises(PausedException) as exc_info:
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                data,
                invalid_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )

        # Should still raise PausedException even with invalid context
        assert isinstance(exc_info.value, PausedException)
        assert str(data) in str(exc_info.value)

    # ============================================================================
    # Phase 1.4: Performance Tests
    # ============================================================================

    async def test_handle_hitl_step_message_performance(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test performance of message generation."""
        import time

        # Arrange
        data = "test_data"
        mock_hitl_step.message_for_user = "Performance test message"

        # Act
        start_time = time.perf_counter()
        with pytest.raises(PausedException):
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )
        end_time = time.perf_counter()

        # Assert - should complete within reasonable time (10ms)
        execution_time = end_time - start_time
        assert execution_time < 0.01

    async def test_handle_hitl_step_context_performance(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test performance of context operations."""
        import time

        # Arrange
        large_data = {"key": "value" * 1000}  # Large data to test performance
        mock_hitl_step.message_for_user = "Large data test"

        # Act
        start_time = time.perf_counter()
        with pytest.raises(PausedException):
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                large_data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )
        end_time = time.perf_counter()

        # Assert - should complete within reasonable time (10ms for large data)
        execution_time = end_time - start_time
        assert execution_time < 0.01

    async def test_handle_hitl_step_memory_usage(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test memory usage patterns."""
        import gc
        import sys

        # Arrange
        data = "test_data"

        # Act
        gc.collect()  # Clean up before test
        initial_memory = sys.getsizeof(mock_context.scratchpad)

        with pytest.raises(PausedException):
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )

        final_memory = sys.getsizeof(mock_context.scratchpad)

        # Assert - memory usage should be reasonable (not excessive)
        memory_increase = final_memory - initial_memory
        assert memory_increase < 1000  # Should not increase by more than 1KB

    async def test_handle_hitl_step_serialization_performance(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test performance of state serialization."""
        import time

        # Arrange
        complex_data = {
            "nested": {"deep": {"structure": {"with": "lots", "of": "data"}}},
            "list": list(range(1000)),
            "string": "x" * 1000,
        }
        mock_hitl_step.message_for_user = "Complex serialization test"

        # Act
        start_time = time.perf_counter()
        with pytest.raises(PausedException):
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                complex_data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )
        end_time = time.perf_counter()

        # Assert - should complete within reasonable time (50ms for complex data)
        execution_time = end_time - start_time
        assert execution_time < 0.05

    # ============================================================================
    # Additional Edge Case Tests
    # ============================================================================

    async def test_handle_hitl_step_none_context(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test HITL step with None context."""
        # Arrange
        data = "test_data"

        # Act & Assert
        with pytest.raises(PausedException) as exc_info:
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                data,
                None,  # None context
                mock_resources,
                mock_limits,
                mock_context_setter,
            )

        # Should still raise PausedException
        assert isinstance(exc_info.value, PausedException)
        assert str(data) in str(exc_info.value)

    async def test_handle_hitl_step_empty_message(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test HITL step with empty message."""
        # Arrange
        data = "test_data"
        mock_hitl_step.message_for_user = ""

        # Act & Assert
        with pytest.raises(PausedException) as exc_info:
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )

        # Should use empty message instead of data
        assert str(exc_info.value) == ""

    async def test_handle_hitl_step_unicode_data(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test HITL step with unicode data."""
        # Arrange
        unicode_data = "测试数据 🚀"
        mock_hitl_step.message_for_user = None

        # Act & Assert
        with pytest.raises(PausedException) as exc_info:
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                unicode_data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )

        # Should handle unicode correctly
        assert unicode_data in str(exc_info.value)

    async def test_handle_hitl_step_binary_data(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test HITL step with binary data."""
        # Arrange
        binary_data = b"binary_data"
        mock_hitl_step.message_for_user = None

        # Act & Assert
        with pytest.raises(PausedException) as exc_info:
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                binary_data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )

        # Should handle binary data correctly
        assert str(binary_data) in str(exc_info.value)
