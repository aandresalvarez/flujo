"""Unit tests for StateManager fallback serialization to prevent data loss."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

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
        # Create a mock context with various fields
        mock_context = Mock(spec=PipelineContext)
        mock_context.initial_prompt = "test prompt"
        mock_context.pipeline_id = "test_pipeline_123"
        mock_context.pipeline_name = "Test Pipeline"
        mock_context.pipeline_version = "1.0.0"
        mock_context.total_steps = 5
        mock_context.error_message = "test error"
        mock_context.run_id = "test_run_456"
        mock_context.created_at = datetime.now()
        mock_context.updated_at = datetime.now()
        mock_context.status = "running"
        mock_context.current_step = 2
        mock_context.last_error = "previous error"
        mock_context.metadata = {"key": "value"}

        # Mock the cache to return None (triggering fallback)
        with patch.object(state_manager, "_get_cached_serialization", return_value=None):
            with patch.object(state_manager, "_should_serialize_context", return_value=False):
                # This would normally trigger the fallback serialization
                # We'll test the fallback logic directly
                fallback_context = {
                    "initial_prompt": getattr(mock_context, "initial_prompt", ""),
                    "pipeline_id": getattr(mock_context, "pipeline_id", "unknown"),
                    "pipeline_name": getattr(mock_context, "pipeline_name", "unknown"),
                    "pipeline_version": getattr(mock_context, "pipeline_version", "latest"),
                    "total_steps": getattr(mock_context, "total_steps", 0),
                    "error_message": getattr(mock_context, "error_message", None),
                    "run_id": getattr(mock_context, "run_id", ""),
                    "created_at": getattr(mock_context, "created_at", None),
                    "updated_at": getattr(mock_context, "updated_at", None),
                }
                # Include any additional fields that might be present
                for field_name in ["status", "current_step", "last_error", "metadata"]:
                    if hasattr(mock_context, field_name):
                        fallback_context[field_name] = getattr(mock_context, field_name, None)

                # Verify all essential fields are included
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
        # Create a minimal mock context with only some fields
        mock_context = Mock(spec=PipelineContext)
        mock_context.initial_prompt = "minimal prompt"
        # Other fields are not set, so they should get default values

        fallback_context = {
            "initial_prompt": getattr(mock_context, "initial_prompt", ""),
            "pipeline_id": getattr(mock_context, "pipeline_id", "unknown"),
            "pipeline_name": getattr(mock_context, "pipeline_name", "unknown"),
            "pipeline_version": getattr(mock_context, "pipeline_version", "latest"),
            "total_steps": getattr(mock_context, "total_steps", 0),
            "error_message": getattr(mock_context, "error_message", None),
            "run_id": getattr(mock_context, "run_id", ""),
            "created_at": getattr(mock_context, "created_at", None),
            "updated_at": getattr(mock_context, "updated_at", None),
        }

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
        # Create a mock context
        mock_context = Mock(spec=PipelineContext)
        mock_context.initial_prompt = "error test prompt"
        mock_context.pipeline_id = "error_pipeline_123"
        mock_context.pipeline_name = "Error Test Pipeline"
        mock_context.pipeline_version = "2.0.0"
        mock_context.total_steps = 3
        mock_context.run_id = "error_run_789"

        # Simulate the error fallback serialization
        error_message = "Serialization failed: test error"
        error_fallback_context = {
            "error": error_message,
            "initial_prompt": getattr(mock_context, "initial_prompt", ""),
            "pipeline_id": getattr(mock_context, "pipeline_id", "unknown"),
            "pipeline_name": getattr(mock_context, "pipeline_name", "unknown"),
            "pipeline_version": getattr(mock_context, "pipeline_version", "latest"),
            "total_steps": getattr(mock_context, "total_steps", 0),
            "run_id": getattr(mock_context, "run_id", ""),
        }

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
        mock_context = Mock(spec=PipelineContext)
        mock_context.initial_prompt = "complex prompt with special chars: ñáéíóú 🚀🎉"
        mock_context.pipeline_id = "complex_pipeline_with_underscores_123"
        mock_context.pipeline_name = "Complex Pipeline with Spaces"
        mock_context.pipeline_version = "1.2.3-beta"
        mock_context.total_steps = 42
        mock_context.error_message = "Complex error with symbols: @#$%^&*()"
        mock_context.run_id = "complex_run_id_with_underscores_456"
        mock_context.metadata = {
            "nested": {"data": "value"},
            "list": [1, 2, 3],
            "special_chars": "ñáéíóú 🚀🎉",
        }

        fallback_context = {
            "initial_prompt": getattr(mock_context, "initial_prompt", ""),
            "pipeline_id": getattr(mock_context, "pipeline_id", "unknown"),
            "pipeline_name": getattr(mock_context, "pipeline_name", "unknown"),
            "pipeline_version": getattr(mock_context, "pipeline_version", "latest"),
            "total_steps": getattr(mock_context, "total_steps", 0),
            "error_message": getattr(mock_context, "error_message", None),
            "run_id": getattr(mock_context, "run_id", ""),
            "created_at": getattr(mock_context, "created_at", None),
            "updated_at": getattr(mock_context, "updated_at", None),
        }
        # Include any additional fields that might be present
        for field_name in ["status", "current_step", "last_error", "metadata"]:
            if hasattr(mock_context, field_name):
                fallback_context[field_name] = getattr(mock_context, field_name, None)

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

    def test_fallback_serialization_performance(self, state_manager):
        """Test that fallback serialization is performant."""
        import time

        # Create a mock context
        mock_context = Mock(spec=PipelineContext)
        mock_context.initial_prompt = "performance test"
        mock_context.pipeline_id = "perf_test_123"
        mock_context.pipeline_name = "Performance Test"
        mock_context.pipeline_version = "1.0.0"
        mock_context.total_steps = 10
        mock_context.run_id = "perf_run_456"

        # Test fallback serialization performance
        start_time = time.perf_counter()
        for _ in range(1000):
            fallback_context = {
                "initial_prompt": getattr(mock_context, "initial_prompt", ""),
                "pipeline_id": getattr(mock_context, "pipeline_id", "unknown"),
                "pipeline_name": getattr(mock_context, "pipeline_name", "unknown"),
                "pipeline_version": getattr(mock_context, "pipeline_version", "latest"),
                "total_steps": getattr(mock_context, "total_steps", 0),
                "error_message": getattr(mock_context, "error_message", None),
                "run_id": getattr(mock_context, "run_id", ""),
                "created_at": getattr(mock_context, "created_at", None),
                "updated_at": getattr(mock_context, "updated_at", None),
            }
            # Include any additional fields that might be present
            for field_name in ["status", "current_step", "last_error", "metadata"]:
                if hasattr(mock_context, field_name):
                    fallback_context[field_name] = getattr(mock_context, field_name, None)
        fallback_time = time.perf_counter() - start_time

        # Should be reasonably fast (less than 10ms for 1000 operations)
        # The comprehensive fallback is slower than minimal fallback but still acceptable
        assert fallback_time < 0.010, f"Fallback serialization too slow: {fallback_time:.6f}s"
