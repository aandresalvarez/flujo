"""
Regression tests for legacy cleanup impact analysis.

This module implements the impact analysis testing strategy outlined in FSD_LEGACY_STEP_LOGIC_CLEANUP.md
to verify that the legacy cleanup was successful and no regressions were introduced.
"""

import inspect
from unittest.mock import Mock, AsyncMock, patch

import pytest

from flujo.application.core.step_logic import (
    _handle_cache_step,
    _handle_hitl_step,
    _run_step_logic,
)
from flujo.application.core.ultra_executor import ExecutorCore
from flujo.domain.dsl.step import HumanInTheLoopStep
from flujo.steps.cache_step import CacheStep
from flujo.domain.models import StepResult
from flujo.exceptions import PausedException


class TestLegacyFunctionUsageAnalysis:
    """Test that legacy functions are properly categorized and handled."""

    async def test_legacy_function_usage_analysis(self):
        """Analyze which legacy functions are still in use."""
        # Verify that migrated functions are no longer in step_logic.py
        import flujo.application.core.step_logic as step_logic

        # These functions should NOT exist (they were migrated)
        assert not hasattr(step_logic, "_execute_loop_step_logic")
        assert not hasattr(step_logic, "_execute_conditional_step_logic")
        assert not hasattr(step_logic, "_execute_parallel_step_logic")
        assert not hasattr(step_logic, "_execute_dynamic_router_step_logic")

        # These functions should exist but be deprecated
        assert hasattr(step_logic, "_handle_cache_step")
        assert hasattr(step_logic, "_handle_hitl_step")
        assert hasattr(step_logic, "_run_step_logic")

        # Verify they are marked as deprecated
        assert hasattr(_handle_cache_step, "__wrapped__")  # Indicates @deprecated_function
        assert hasattr(_handle_hitl_step, "__wrapped__")
        assert hasattr(_run_step_logic, "__wrapped__")

    async def test_import_dependency_analysis(self):
        """Analyze import dependencies on legacy functions."""
        # Test that new handler functions delegate to ExecutorCore
        with patch("flujo.application.core.ultra_executor.ExecutorCore") as mock_executor_class:
            mock_executor = Mock()
            mock_executor_class.return_value = mock_executor

            # Test _handle_loop_step delegation
            from flujo.application.core.step_logic import _handle_loop_step

            mock_executor._handle_loop_step = AsyncMock(return_value=StepResult(name="test"))

            result = await _handle_loop_step(
                step=Mock(),
                data="test",
                context=None,
                resources=None,
                step_executor=AsyncMock(),
                context_model_defined=True,
                usage_limits=None,
                context_setter=Mock(),
            )

            mock_executor._handle_loop_step.assert_called_once()
            assert isinstance(result, StepResult)

    async def test_backward_compatibility_verification(self):
        """Verify that remaining legacy functions work correctly."""
        # Test that deprecated functions still work
        mock_cache_step = Mock(spec=CacheStep)
        mock_cache_step.wrapped_step = Mock()
        mock_cache_step.wrapped_step.name = "test_step"
        mock_cache_step.wrapped_step.agent = None
        mock_cache_step.wrapped_step.config = Mock()
        mock_cache_step.wrapped_step.config.max_retries = 1
        mock_cache_step.wrapped_step.config.timeout_s = 30
        mock_cache_step.wrapped_step.config.temperature = None
        mock_cache_step.wrapped_step.plugins = []
        mock_cache_step.wrapped_step.validators = []
        mock_cache_step.wrapped_step.processors = Mock()
        mock_cache_step.wrapped_step.processors.prompt_processors = []
        mock_cache_step.wrapped_step.processors.output_processors = []
        mock_cache_step.wrapped_step.updates_context = False
        mock_cache_step.wrapped_step.persist_feedback_to_context = None
        mock_cache_step.wrapped_step.persist_validation_results_to = None
        mock_cache_step.cache_backend = Mock()
        mock_cache_step.cache_backend.get.return_value = None

        mock_step_executor = AsyncMock()
        mock_step_executor.return_value = StepResult(name="test", success=True)

        # This should not raise an exception despite being deprecated
        result = await _handle_cache_step(
            step=mock_cache_step,
            data="test",
            context=None,
            resources=None,
            step_executor=mock_step_executor,
        )

        assert isinstance(result, StepResult)


class TestMigrationCompleteness:
    """Test that migration is complete and safe."""

    async def test_migrated_functions_removal(self):
        """Test that migrated functions can be safely removed."""
        # Verify that the old function names are not accessible
        import flujo.application.core.step_logic as step_logic

        # These should not exist
        with pytest.raises(AttributeError):
            getattr(step_logic, "_execute_loop_step_logic")

        with pytest.raises(AttributeError):
            getattr(step_logic, "_execute_conditional_step_logic")

        with pytest.raises(AttributeError):
            getattr(step_logic, "_execute_parallel_step_logic")

        with pytest.raises(AttributeError):
            getattr(step_logic, "_execute_dynamic_router_step_logic")

    async def test_legacy_function_deprecation(self):
        """Test deprecation warnings for remaining legacy functions."""
        # Test that deprecated functions emit warnings
        mock_cache_step = Mock(spec=CacheStep)
        mock_cache_step.wrapped_step = Mock()
        mock_cache_step.wrapped_step.name = "test_step"
        mock_cache_step.wrapped_step.agent = None
        mock_cache_step.wrapped_step.config = Mock()
        mock_cache_step.wrapped_step.config.max_retries = 1
        mock_cache_step.wrapped_step.config.timeout_s = 30
        mock_cache_step.wrapped_step.config.temperature = None
        mock_cache_step.wrapped_step.plugins = []
        mock_cache_step.wrapped_step.validators = []
        mock_cache_step.wrapped_step.processors = Mock()
        mock_cache_step.wrapped_step.processors.prompt_processors = []
        mock_cache_step.wrapped_step.processors.output_processors = []
        mock_cache_step.wrapped_step.updates_context = False
        mock_cache_step.wrapped_step.persist_feedback_to_context = None
        mock_cache_step.wrapped_step.persist_validation_results_to = None
        mock_cache_step.cache_backend = Mock()
        mock_cache_step.cache_backend.get.return_value = None

        with pytest.warns(DeprecationWarning, match="is deprecated"):
            await _handle_cache_step(
                step=mock_cache_step,
                data="test",
                context=None,
                resources=None,
                step_executor=AsyncMock(),
            )

        # HITL step: test warning and exception separately
        mock_hitl_step = Mock(spec=HumanInTheLoopStep)
        mock_hitl_step.message_for_user = "Test message"
        # Test warning
        with pytest.warns(DeprecationWarning, match="is deprecated"):
            try:
                await _handle_hitl_step(
                    step=mock_hitl_step,
                    data="test",
                    context=None,
                )
            except PausedException:
                pass
        # Test exception
        with pytest.raises(PausedException, match="Test message"):
            await _handle_hitl_step(
                step=mock_hitl_step,
                data="test",
                context=None,
            )

        # _run_step_logic: use a real config object, not a Mock
        class DummyConfig:
            max_retries = 1
            timeout_s = 30
            temperature = None

        mock_step = Mock()
        mock_step.name = "test_step"
        mock_step.agent = Mock()
        mock_step.agent.run = AsyncMock(return_value="test_output")
        mock_step.config = DummyConfig()
        mock_step.processors = Mock()
        mock_step.processors.prompt_processors = []
        mock_step.processors.output_processors = []
        mock_step.plugins = []
        mock_step.validators = []
        mock_step.failure_handlers = []
        mock_step.fallback_step = None
        mock_step.persist_feedback_to_context = None

        with pytest.warns(DeprecationWarning, match="is deprecated"):
            await _run_step_logic(
                step=mock_step,
                data="test",
                context=None,
                resources=None,
                step_executor=AsyncMock(),
                context_model_defined=True,
            )

    async def test_import_path_updates(self):
        """Test that import paths are updated correctly."""
        # Verify that ExecutorCore has the migrated methods
        executor = ExecutorCore()

        assert hasattr(executor, "_handle_loop_step")
        assert hasattr(executor, "_handle_conditional_step")
        assert hasattr(executor, "_handle_parallel_step")
        assert hasattr(executor, "_handle_dynamic_router_step")

        # Verify they are callable
        assert callable(executor._handle_loop_step)
        assert callable(executor._handle_conditional_step)
        assert callable(executor._handle_parallel_step)
        assert callable(executor._handle_dynamic_router_step)


class TestDeprecationDecorator:
    """Test the deprecation decorator functionality."""

    def test_deprecated_function_decorator(self):
        """Test that the deprecated_function decorator works correctly."""
        from flujo.application.core.step_logic import deprecated_function

        @deprecated_function
        def test_function():
            return "test"

        # Test that the decorator preserves the function signature
        assert test_function.__name__ == "test_function"
        assert test_function.__wrapped__ is not None

    async def test_deprecation_warning_message(self):
        """Test that deprecation warnings have the correct message."""
        mock_cache_step = Mock(spec=CacheStep)
        mock_cache_step.wrapped_step = Mock()
        mock_cache_step.wrapped_step.name = "test_step"
        mock_cache_step.wrapped_step.agent = None
        mock_cache_step.wrapped_step.config = Mock()
        mock_cache_step.wrapped_step.config.max_retries = 1
        mock_cache_step.wrapped_step.config.timeout_s = 30
        mock_cache_step.wrapped_step.config.temperature = None
        mock_cache_step.wrapped_step.plugins = []
        mock_cache_step.wrapped_step.validators = []
        mock_cache_step.wrapped_step.processors = Mock()
        mock_cache_step.wrapped_step.processors.prompt_processors = []
        mock_cache_step.wrapped_step.processors.output_processors = []
        mock_cache_step.wrapped_step.updates_context = False
        mock_cache_step.wrapped_step.persist_feedback_to_context = None
        mock_cache_step.wrapped_step.persist_validation_results_to = None
        mock_cache_step.cache_backend = Mock()
        mock_cache_step.cache_backend.get.return_value = None

        with pytest.warns(DeprecationWarning) as record:
            await _handle_cache_step(
                step=mock_cache_step,
                data="test",
                context=None,
                resources=None,
                step_executor=AsyncMock(),
            )

        # Check that the warning message contains the expected text
        assert any(
            "is deprecated and will be removed" in str(warning.message) for warning in record
        )


class TestFunctionSignatureAnalysis:
    """Test that function signatures are preserved correctly."""

    def test_deprecated_function_signatures(self):
        """Test that deprecated functions maintain their original signatures."""
        # Test _handle_cache_step signature
        sig = inspect.signature(_handle_cache_step)
        params = list(sig.parameters.keys())

        expected_params = ["step", "data", "context", "resources", "step_executor"]
        assert params == expected_params

        # Test _handle_hitl_step signature
        sig = inspect.signature(_handle_hitl_step)
        params = list(sig.parameters.keys())

        expected_params = ["step", "data", "context"]
        assert params == expected_params

        # Test _run_step_logic signature
        sig = inspect.signature(_run_step_logic)
        params = list(sig.parameters.keys())

        # Should have the expected parameters (including keyword-only ones)
        assert "step" in params
        assert "data" in params
        assert "context" in params
        assert "resources" in params
        assert "step_executor" in params


class TestLegacyCleanupCompleteness:
    """Test that the legacy cleanup is complete and comprehensive."""

    def test_no_orphaned_imports(self):
        """Test that there are no orphaned imports from removed functions."""
        import flujo.application.core.step_logic as step_logic

        # Check that imports for migrated step types are still needed
        # (they might be used by the new handler functions)
        assert hasattr(step_logic, "LoopStep")
        assert hasattr(step_logic, "ConditionalStep")
        assert hasattr(step_logic, "ParallelStep")
        assert hasattr(step_logic, "DynamicParallelRouterStep")

        # These should still be imported for the remaining functions
        assert hasattr(step_logic, "HumanInTheLoopStep")
        assert hasattr(step_logic, "CacheStep")

    def test_legacy_function_comments(self):
        """Test that removal comments are present in the code."""
        # This test verifies that the cleanup was documented in the code
        with open("flujo/application/core/step_logic.py", "r") as f:
            content = f.read()

        # Should have comments indicating removed functions
        assert "# _execute_parallel_step_logic removed" in content
        assert "# _execute_loop_step_logic removed" in content
        assert "# _execute_conditional_step_logic removed" in content
        assert "# _execute_dynamic_router_step_logic removed" in content

    def test_new_handler_functions_exist(self):
        """Test that the new handler functions exist and work."""
        from flujo.application.core.step_logic import (
            _handle_loop_step,
            _handle_conditional_step,
            _handle_parallel_step,
            _handle_dynamic_router_step,
        )

        # All new handler functions should exist
        assert callable(_handle_loop_step)
        assert callable(_handle_conditional_step)
        assert callable(_handle_parallel_step)
        assert callable(_handle_dynamic_router_step)

        # They should not be deprecated (they're the new implementations)
        assert not hasattr(_handle_loop_step, "__wrapped__")
        assert not hasattr(_handle_conditional_step, "__wrapped__")
        assert not hasattr(_handle_parallel_step, "__wrapped__")
        assert not hasattr(_handle_dynamic_router_step, "__wrapped__")
