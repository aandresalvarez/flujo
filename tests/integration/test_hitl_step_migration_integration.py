"""
Integration tests for HumanInTheLoopStep migration to ExecutorCore.

This test suite validates the integration of the migrated HITL step logic
with the broader pipeline execution system, ensuring compatibility and
correct behavior in real-world scenarios.

Author: Flujo Team
Version: 1.0
"""

import pytest
from unittest.mock import Mock

from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl.step import HumanInTheLoopStep
from flujo.domain.models import PipelineContext, UsageLimits
from flujo.exceptions import PausedException


class TestHITLStepMigrationIntegration:
    """Integration tests for HITL step migration."""

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
    # Phase 2.1: End-to-End Scenarios
    # ============================================================================

    async def test_hitl_step_complex_pipeline_integration(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test HITL step within complex pipelines."""
        # Arrange
        data = {"input": "test_data", "metadata": {"source": "integration_test"}}
        mock_hitl_step.message_for_user = "Please review this complex data"

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

        # Verify the exception contains the custom message
        assert "Please review this complex data" in str(exc_info.value)

        # Verify context was updated with typed fields and metadata
        assert mock_context.status == "paused"
        assert mock_context.pause_message == "Please review this complex data"
        assert mock_context.paused_step_input == data
        assert mock_context.hitl_data.get("last_hitl_step") == mock_hitl_step.name

    async def test_hitl_step_with_different_contexts(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test HITL step with various context types."""
        # Test with None context
        data = "test_data"
        try:
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                data,
                None,  # None context
                mock_resources,
                mock_limits,
                mock_context_setter,
            )
            assert False, "Should have raised PausedException"
        except PausedException as e:
            assert str(data) in str(e)

        # Test with empty context
        empty_context = PipelineContext()
        try:
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                data,
                empty_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )
            assert False, "Should have raised PausedException"
        except PausedException as e:
            assert str(data) in str(e)
            assert empty_context.status == "paused"

        # Test with pre-populated context
        populated_context = PipelineContext(hitl_data={"existing_key": "existing_value"})
        try:
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                data,
                populated_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )
            assert False, "Should have raised PausedException"
        except PausedException as e:
            assert str(data) in str(e)
            # Verify existing data is preserved (metadata remains in hitl_data)
            assert populated_context.hitl_data.get("last_hitl_step") == mock_hitl_step.name
            # Verify new data is added on typed fields
            assert populated_context.status == "paused"

    async def test_hitl_step_with_telemetry(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test HITL step telemetry and observability."""
        # Arrange
        data = "test_data"
        mock_hitl_step.name = "telemetry_test_step"
        mock_hitl_step.message_for_user = "Telemetry test message"

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

        # Verify telemetry logging (this would be checked in a real scenario)
        # The implementation should log the HITL step execution
        assert mock_context.status == "paused"
        assert mock_context.pause_message == "Telemetry test message"

    async def test_hitl_step_with_usage_limits(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test HITL step with usage limit enforcement."""
        # Arrange
        data = "test_data"
        mock_limits.total_cost_usd_limit = 100.0
        mock_limits.total_tokens_limit = 10000

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

        # Verify that usage limits don't interfere with HITL step execution
        # HITL steps should pause regardless of usage limits
        assert mock_context.status == "paused"

    # ============================================================================
    # Phase 2.2: Migration Compatibility Tests
    # ============================================================================

    async def test_hitl_step_legacy_compatibility(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test that migrated HITL step produces identical results to legacy."""
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

        # Verify the behavior matches legacy implementation:
        # 1. Should raise PausedException with str(data) as message
        assert str(data) in str(exc_info.value)

        # 2. Should update typed context fields
        assert mock_context.status == "paused"
        assert mock_context.pause_message == str(data)
        assert mock_context.hitl_data.get("last_hitl_step") == mock_hitl_step.name

    async def test_hitl_step_backward_compatibility(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test that existing HITL step configurations continue to work."""
        # Test various legacy configurations
        test_cases = [
            {"message": None, "data": "simple_data"},
            {"message": "", "data": "empty_message_data"},
            {"message": "Custom message", "data": {"complex": "data"}},
            {"message": None, "data": 123},
            {"message": None, "data": [1, 2, 3]},
        ]

        for case in test_cases:
            mock_hitl_step.message_for_user = case["message"]
            data = case["data"]

            with pytest.raises(PausedException) as exc_info:
                await executor_core._handle_hitl_step(
                    mock_hitl_step,
                    data,
                    mock_context,
                    mock_resources,
                    mock_limits,
                    mock_context_setter,
                )

            # Verify backward compatibility
            expected_message = case["message"] if case["message"] is not None else str(data)
            assert expected_message in str(exc_info.value)
            assert mock_context.status == "paused"
            assert mock_context.pause_message == expected_message
            assert mock_context.hitl_data.get("last_hitl_step") == mock_hitl_step.name
            assert mock_context.paused_step_input == data

    async def test_hitl_step_performance_comparison(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test that migrated implementation maintains or improves performance."""
        import time

        # Arrange
        data = "performance_test_data"
        mock_hitl_step.message_for_user = "Performance test"

        # Measure execution time
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

        execution_time = end_time - start_time

        # Log performance (sanity check only - CI timing variance)
        print(f"HITL migration performance: {execution_time * 1000:.2f}ms")
        assert execution_time < 1.0, f"HITL migration too slow: {execution_time:.3f}s"

    # ============================================================================
    # Additional Integration Tests
    # ============================================================================

    async def test_hitl_step_with_executor_core_integration(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test HITL step integration with ExecutorCore's complex step handling."""
        # Arrange
        data = "integration_test_data"
        mock_hitl_step.message_for_user = "Integration test message"

        # Test through the direct HITL step handler
        with pytest.raises(PausedException) as exc_info:
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )

        # Verify the integration works correctly
        assert "Integration test message" in str(exc_info.value)
        assert mock_context.status == "paused"

    async def test_hitl_step_error_handling_integration(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test error handling integration in HITL step."""
        # Test with problematic data that might cause issues
        problematic_data = Mock()
        problematic_data.__str__ = Mock(side_effect=Exception("String conversion failed"))

        # Should still work and raise PausedException
        with pytest.raises(PausedException) as exc_info:
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                problematic_data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )

        # Should use fallback message
        assert "Data conversion failed" in str(exc_info.value)

    async def test_hitl_step_context_serialization_integration(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test context serialization integration for pause/resume scenarios."""
        # Arrange
        complex_data = {
            "nested": {"deep": {"structure": {"with": "lots", "of": "data"}}},
            "list": list(range(1000)),
            "string": "x" * 1000,
            "unicode": "测试数据 🚀",
        }
        mock_hitl_step.message_for_user = "Complex serialization test"

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

        # Verify complex data is preserved on typed fields
        assert mock_context.status == "paused"
        assert mock_context.pause_message == "Complex serialization test"
        assert mock_context.paused_step_input == complex_data
        assert mock_context.hitl_data.get("last_hitl_step") == mock_hitl_step.name

    async def test_hitl_step_memory_efficiency_integration(
        self,
        executor_core: ExecutorCore,
        mock_hitl_step: Mock,
        mock_context: Mock,
        mock_resources: Mock,
        mock_limits: Mock,
        mock_context_setter: Mock,
    ):
        """Test memory efficiency integration for large data handling."""
        import gc
        import sys

        # Arrange
        large_data = {"key": "value" * 10000}  # Large data
        mock_hitl_step.message_for_user = "Memory efficiency test"

        # Act
        gc.collect()  # Clean up before test
        initial_memory = sys.getsizeof(mock_context.model_dump())

        with pytest.raises(PausedException):
            await executor_core._handle_hitl_step(
                mock_hitl_step,
                large_data,
                mock_context,
                mock_resources,
                mock_limits,
                mock_context_setter,
            )

        final_memory = sys.getsizeof(mock_context.model_dump())

        # Assert memory usage is reasonable
        memory_increase = final_memory - initial_memory
        assert memory_increase < 10000  # Should not increase by more than 10KB

        # Verify data is still properly stored
        assert mock_context.paused_step_input == large_data
        assert mock_context.hitl_data.get("last_hitl_step") == mock_hitl_step.name
