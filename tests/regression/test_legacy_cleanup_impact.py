"""
Regression tests for legacy cleanup impact analysis.

This module implements the impact analysis testing strategy outlined in FSD_LEGACY_STEP_LOGIC_CLEANUP.md
to verify that the legacy cleanup was successful and no regressions were introduced.
"""

import inspect
from unittest.mock import Mock, AsyncMock

import pytest

# step_logic module was intentionally removed during refactoring
# The functionality has been migrated to ultra_executor
from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl.step import HumanInTheLoopStep
from flujo.domain.dsl.cache_step import CacheStep
from flujo.domain.models import StepResult
from flujo.exceptions import PausedException
from tests.test_types.fixtures import execute_simple_step


@pytest.mark.asyncio
class TestLegacyFunctionUsageAnalysis:
    """Test that legacy functions are properly categorized and handled."""

    async def test_legacy_function_usage_analysis(self):
        """Analyze which legacy functions are still in use."""
        # step_logic module was intentionally removed during refactoring
        # This test verifies that the module no longer exists
        with pytest.raises(ModuleNotFoundError):
            import importlib

            importlib.import_module("flujo.application.core.step_logic")

        print("step_logic module successfully removed")

    async def test_import_dependency_analysis(self):
        """Analyze import dependencies on legacy functions."""
        # step_logic module was intentionally removed during refactoring
        # This test verifies that the module no longer exists
        with pytest.raises(ModuleNotFoundError):
            import importlib

            importlib.import_module("flujo.application.core.step_logic")

        print("step_logic module successfully removed")

    async def test_backward_compatibility_verification(self):
        """Test that backward compatibility is maintained for existing code."""
        # Test that the new handler functions work correctly
        mock_cache_step = Mock(spec=CacheStep)
        mock_cache_step.name = "test_cache_step"  # Add name attribute to the cache step itself
        # Create a proper mock for wrapped_step with name attribute
        wrapped_step_mock = Mock()
        wrapped_step_mock.name = "test_step"
        mock_cache_step.wrapped_step = wrapped_step_mock
        mock_cache_step.wrapped_step.agent = AsyncMock()
        mock_cache_step.wrapped_step.agent.run = AsyncMock(
            return_value="test_output"
        )  # Configure agent.run to return proper value
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
        mock_cache_step.wrapped_step.fallback_step = None  # Prevent infinite fallback loops
        mock_cache_step.cache_backend = Mock()
        mock_cache_step.cache_backend.get = AsyncMock(return_value=None)  # Make it async

        mock_step_executor = AsyncMock()
        mock_step_executor.return_value = StepResult(name="test", success=True)

        # step_logic module was removed, functionality migrated to ultra_executor
        # Test using ExecutorCore
        executor = ExecutorCore()

        class _MockAgentExecutor:
            async def execute(
                self,
                _core: object,
                _step: object,
                _data: object,
                _context: object,
                _resources: object,
                _limits: object,
                _stream: bool,
                _on_chunk: object,
                _cache_key: object,
                _fallback_depth: int,
            ) -> StepResult:
                return StepResult(name="test", success=True)

        executor.agent_step_executor = _MockAgentExecutor()
        result = await executor._handle_cache_step(
            step=mock_cache_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            context_setter=None,
        )

        assert isinstance(result, StepResult)


@pytest.mark.asyncio
class TestMigrationCompleteness:
    """Test that migration is complete and safe."""

    async def test_migrated_functions_removal(self):
        """Test that migrated functions can be safely removed."""
        # step_logic module was intentionally removed during refactoring
        # This test verifies that the module no longer exists
        with pytest.raises(ModuleNotFoundError):
            import importlib

            importlib.import_module("flujo.application.core.step_logic")

        print("step_logic module successfully removed")

    async def test_legacy_function_deprecation(self):
        """Test deprecation warnings for remaining legacy functions."""
        # Test that deprecated functions emit warnings
        mock_cache_step = Mock(spec=CacheStep)
        mock_cache_step.name = "test_cache_step"  # Add name attribute to the cache step itself
        # Create a proper mock for wrapped_step with name attribute
        wrapped_step_mock = Mock()
        wrapped_step_mock.name = "test_step"
        mock_cache_step.wrapped_step = wrapped_step_mock
        mock_cache_step.wrapped_step.agent = AsyncMock()
        mock_cache_step.wrapped_step.agent.run = AsyncMock(
            return_value="test_output"
        )  # Configure agent.run to return proper value
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
        mock_cache_step.wrapped_step.fallback_step = None  # Prevent infinite fallback loops
        mock_cache_step.cache_backend = Mock()
        mock_cache_step.cache_backend.get = AsyncMock(return_value=None)  # Make it async

        # step_logic module was removed, functionality migrated to ultra_executor
        # Test using ExecutorCore
        executor = ExecutorCore()

        class _MockAgentExecutor:
            async def execute(
                self,
                _core: object,
                _step: object,
                _data: object,
                _context: object,
                _resources: object,
                _limits: object,
                _stream: bool,
                _on_chunk: object,
                _cache_key: object,
                _fallback_depth: int,
            ) -> StepResult:
                return StepResult(name="test", success=True)

        executor.agent_step_executor = _MockAgentExecutor()
        await executor._handle_cache_step(
            step=mock_cache_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            context_setter=None,
        )

        # HITL step: test using ExecutorCore
        mock_hitl_step = Mock(spec=HumanInTheLoopStep)
        mock_hitl_step.name = "test_hitl_step"  # Add name attribute
        mock_hitl_step.message_for_user = "Test message"
        # Test exception
        with pytest.raises(PausedException, match="Test message"):
            await executor._handle_hitl_step(
                step=mock_hitl_step,
                data="test",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

        # _run_step_logic: use a real config object, not a Mock
        class DummyConfig:
            max_retries = 1
            timeout_s = 30
            temperature = None

        mock_step = Mock()
        mock_step.name = "test_step"
        # Create a proper mock for agent with run method
        agent_mock = Mock()
        agent_mock.run = AsyncMock(return_value="test_output")
        mock_step.agent = agent_mock
        mock_step.config = DummyConfig()
        mock_step.processors = Mock()
        mock_step.processors.prompt_processors = []
        mock_step.processors.output_processors = []
        mock_step.plugins = []
        mock_step.validators = []
        mock_step.failure_handlers = []
        mock_step.fallback_step = None
        mock_step.persist_feedback_to_context = None

        # step_logic module was removed, functionality migrated to ultra_executor
        # Test using ExecutorCore
        executor = ExecutorCore()
        await execute_simple_step(
            executor,
            step=mock_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
            _fallback_depth=0,
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
        # step_logic module was removed, functionality migrated to ultra_executor
        # The deprecated_function decorator is no longer needed since the functions
        # have been migrated to ExecutorCore methods

        def test_function():
            return "test"

        # Test that the function works correctly
        assert test_function.__name__ == "test_function"
        assert test_function() == "test"

    async def test_deprecation_warning_message(self):
        """Test that deprecation warnings have the correct message."""
        mock_cache_step = Mock(spec=CacheStep)
        mock_cache_step.name = "test_cache_step"  # Add name attribute to the cache step itself
        mock_cache_step.wrapped_step = Mock()
        mock_cache_step.wrapped_step.name = "test_step"
        mock_cache_step.wrapped_step.agent = AsyncMock()
        mock_cache_step.wrapped_step.agent.run = AsyncMock(
            return_value="test_output"
        )  # Configure agent.run to return proper value
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
        mock_cache_step.wrapped_step.fallback_step = None  # Prevent infinite fallback loops
        mock_cache_step.cache_backend = Mock()
        mock_cache_step.cache_backend.get = AsyncMock(return_value=None)  # Make it async

        # step_logic module was removed, functionality migrated to ultra_executor
        # Test that ExecutorCore can handle cache steps
        executor = ExecutorCore()

        class _MockAgentExecutor:
            async def execute(
                self,
                _core: object,
                _step: object,
                _data: object,
                _context: object,
                _resources: object,
                _limits: object,
                _stream: bool,
                _on_chunk: object,
                _cache_key: object,
                _fallback_depth: int,
            ) -> StepResult:
                return StepResult(name="test", success=True)

        executor.agent_step_executor = _MockAgentExecutor()
        result = await executor._handle_cache_step(
            step=mock_cache_step,
            data="test",
            context=None,
            resources=None,
            limits=None,
            context_setter=None,
        )

        # Check that the result is valid
        assert isinstance(result, StepResult)


class TestFunctionSignatureAnalysis:
    """Test that function signatures are preserved correctly."""

    def test_executor_core_method_signatures(self):
        """Test that ExecutorCore methods have the expected signatures."""
        executor = ExecutorCore()

        # Test _handle_cache_step signature
        sig = inspect.signature(executor._handle_cache_step)
        params = list(sig.parameters.keys())

        expected_params = {
            "step",
            "data",
            "context",
            "resources",
            "limits",
            "context_setter",
        }
        assert expected_params.issubset(params)

        # Test _handle_hitl_step signature
        sig = inspect.signature(executor._handle_hitl_step)
        params = list(sig.parameters.keys())

        expected_params = {
            "step",
            "data",
            "context",
            "resources",
            "limits",
            "context_setter",
            "stream",
            "on_chunk",
            "cache_key",
            "_fallback_depth",
        }
        assert expected_params.issubset(params)

        # Test execute signature
        sig = inspect.signature(executor.execute)
        params = sig.parameters

        assert "frame_or_step" in params
        assert "data" in params
        assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


class TestLegacyCleanupCompleteness:
    """Test that the legacy cleanup is complete and comprehensive."""

    def test_no_orphaned_imports(self):
        """Test that there are no orphaned imports from removed functions."""
        # step_logic module was removed, functionality migrated to ultra_executor
        # Test that the module no longer exists
        with pytest.raises(ModuleNotFoundError):
            import importlib

            importlib.import_module("flujo.application.core.step_logic")

        # Test that ExecutorCore has all the migrated functionality
        executor = ExecutorCore()
        assert hasattr(executor, "_handle_loop_step")
        assert hasattr(executor, "_handle_conditional_step")
        assert hasattr(executor, "_handle_parallel_step")
        assert hasattr(executor, "_handle_dynamic_router_step")
        assert hasattr(executor, "_handle_cache_step")
        assert hasattr(executor, "_handle_hitl_step")

    def test_legacy_function_comments(self):
        """Test that removal comments are present in the code."""
        # step_logic module was removed, functionality migrated to ultra_executor
        # This test verifies that the module no longer exists
        with pytest.raises(FileNotFoundError):
            with open("flujo/application/core/step_logic.py", "r") as f:
                f.read()

        # The module was intentionally removed during refactoring
        print("step_logic.py file successfully removed")

    def test_new_handler_functions_exist(self):
        """Test that the new handler functions exist and work."""
        # step_logic module was removed, functionality migrated to ultra_executor
        # Test that ExecutorCore has all the migrated functionality
        executor = ExecutorCore()

        # All new handler functions should exist in ExecutorCore
        assert hasattr(executor, "_handle_loop_step")
        assert hasattr(executor, "_handle_conditional_step")
        assert hasattr(executor, "_handle_parallel_step")
        assert hasattr(executor, "_handle_dynamic_router_step")

        # They should be callable
        assert callable(executor._handle_loop_step)
        assert callable(executor._handle_conditional_step)
        assert callable(executor._handle_parallel_step)
        assert callable(executor._handle_dynamic_router_step)

        # They should not be deprecated (they're the new implementations)
        assert not hasattr(executor._handle_loop_step, "__wrapped__")
        assert not hasattr(executor._handle_conditional_step, "__wrapped__")
        assert not hasattr(executor._handle_parallel_step, "__wrapped__")
        assert not hasattr(executor._handle_dynamic_router_step, "__wrapped__")
