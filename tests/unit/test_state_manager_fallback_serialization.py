"""Unit tests for StateManager fallback serialization to prevent data loss."""

import pytest
from datetime import datetime

from flujo.application.core.state_manager import StateManager
from flujo.domain.models import PipelineContext


class TestStateManagerFallbackSerialization:
    """Test fallback serialization to prevent data loss."""

    @pytest.fixture
    def state_manager(self):
        """Create a StateManager instance for testing."""
        return StateManager()

    def test_fallback_serialization_includes_all_essential_fields(self, state_manager):
        """Test that fallback serialization includes all essential context fields."""
        # Create a real context with various fields
        context = PipelineContext(initial_prompt="test prompt")
        context.pipeline_id = "test_pipeline_123"
        context.pipeline_name = "Test Pipeline"
        context.pipeline_version = "1.0.0"
        context.total_steps = 5
        context.error_message = "test error"
        context.run_id = "test_run_456"
        context.created_at = datetime.now()
        context.updated_at = datetime.now()
        context.status = "running"
        context.current_step = 2
        context.last_error = "previous error"
        context.metadata = {"key": "value"}

        fallback_context = state_manager._build_context_fallback(context)

        assert fallback_context["initial_prompt"] == "test prompt"
        assert fallback_context["pipeline_id"] == "test_pipeline_123"
        assert fallback_context["pipeline_name"] == "Test Pipeline"
        assert fallback_context["pipeline_version"] == "1.0.0"
        assert fallback_context["total_steps"] == 5
        assert fallback_context["error_message"] == "test error"
        assert fallback_context["run_id"] == "test_run_456"
        assert fallback_context["status"] == "running"
        assert fallback_context["current_step"] == 2
        assert fallback_context["last_error"] == "previous error"
        assert fallback_context["metadata"] == {"key": "value"}

    def test_fallback_serialization_with_missing_fields(self, state_manager):
        """Test that fallback serialization handles missing fields gracefully."""

        # Minimal context object with only the fields under test
        class MinimalContext:
            def __init__(self, initial_prompt: str) -> None:
                self.initial_prompt = initial_prompt

        context = MinimalContext("minimal prompt")
        # Other fields are not set, so they should get default values

        fallback_context = state_manager._build_context_fallback(context)

        # Verify default values are used for missing fields
        assert fallback_context["initial_prompt"] == "minimal prompt"
        assert fallback_context["pipeline_id"] == "unknown"
        assert fallback_context["pipeline_name"] == "unknown"
        assert fallback_context["pipeline_version"] == "latest"
        assert fallback_context["total_steps"] == 0
        assert fallback_context["error_message"] is None
        assert fallback_context["run_id"] == ""

    def test_error_fallback_serialization(self, state_manager):
        """Test that error fallback serialization includes essential fields."""
        # Create a real context
        context = PipelineContext(initial_prompt="error test prompt")
        context.pipeline_id = "error_pipeline_123"
        context.pipeline_name = "Error Test Pipeline"
        context.pipeline_version = "2.0.0"
        context.total_steps = 3
        context.run_id = "error_run_789"

        # Simulate the error fallback serialization
        error_message = "Serialization failed: test error"
        error_fallback_context = state_manager._build_context_fallback(
            context, error_message=error_message
        )

        # Verify error fallback includes essential fields
        assert error_fallback_context["error"] == error_message
        assert error_fallback_context["initial_prompt"] == "error test prompt"
        assert error_fallback_context["pipeline_id"] == "error_pipeline_123"
        assert error_fallback_context["pipeline_name"] == "Error Test Pipeline"
        assert error_fallback_context["pipeline_version"] == "2.0.0"
        assert error_fallback_context["total_steps"] == 3
        assert error_fallback_context["run_id"] == "error_run_789"

    def test_fallback_serialization_preserves_data_integrity(self, state_manager):
        """Test that fallback serialization preserves data integrity."""
        # Create a context with complex data
        context = PipelineContext(initial_prompt="complex prompt with special chars: ñáéíóú 🚀🎉")
        context.pipeline_id = "complex_pipeline_with_underscores_123"
        context.pipeline_name = "Complex Pipeline with Spaces"
        context.pipeline_version = "1.2.3-beta"
        context.total_steps = 42
        context.error_message = "Complex error with symbols: @#$%^&*()"
        context.run_id = "complex_run_id_with_underscores_456"
        context.metadata = {
            "nested": {"data": "value"},
            "list": [1, 2, 3],
            "special_chars": "ñáéíóú 🚀🎉",
        }

        fallback_context = state_manager._build_context_fallback(context)

        # Verify data integrity is preserved
        assert (
            fallback_context["initial_prompt"] == "complex prompt with special chars: ñáéíóú 🚀🎉"
        )
        assert fallback_context["pipeline_id"] == "complex_pipeline_with_underscores_123"
        assert fallback_context["pipeline_name"] == "Complex Pipeline with Spaces"
        assert fallback_context["pipeline_version"] == "1.2.3-beta"
        assert fallback_context["total_steps"] == 42
        assert fallback_context["error_message"] == "Complex error with symbols: @#$%^&*()"
        assert fallback_context["run_id"] == "complex_run_id_with_underscores_456"
        assert fallback_context["metadata"] == {
            "nested": {"data": "value"},
            "list": [1, 2, 3],
            "special_chars": "ñáéíóú 🚀🎉",
        }
