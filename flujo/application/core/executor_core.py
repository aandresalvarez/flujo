"""
Executor core: policy-driven step executor extracted from ultra_executor.

This module defines the `ExecutorCore` and error types that callers depend on.
Concrete defaults live in `default_components.py`; protocols in
`executor_protocols.py`.
"""

from __future__ import annotations

import asyncio
import warnings
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Generic, Optional, cast
from pydantic import BaseModel

from ...domain.dsl.conditional import ConditionalStep
from ...domain.dsl.dynamic_router import DynamicParallelRouterStep
from ...domain.dsl.loop import LoopStep
from ...domain.dsl.parallel import ParallelStep
from ...domain.dsl.step import HumanInTheLoopStep, Step
from ...domain.interfaces import StateProvider
from ...domain.models import Failure, PipelineResult, Quota, StepOutcome, StepResult, UsageLimits
from ...exceptions import (
    InfiniteFallbackError,
    UsageLimitExceededError,
    PausedException,
    MissingAgentError,
    MockDetectionError,
    InfiniteRedirectError,
    PipelineAbortSignal,
)
from .quota_manager import QuotaManager
from .fallback_handler import FallbackHandler
from .background_task_manager import BackgroundTaskManager
from .cache_manager import CacheManager
from .hydration_manager import HydrationManager
from .execution_dispatcher import ExecutionDispatcher
from .agent_orchestrator import AgentOrchestrator
from .conditional_orchestrator import ConditionalOrchestrator
from .hitl_orchestrator import HitlOrchestrator
from .loop_orchestrator import LoopOrchestrator
from .failure_builder import build_failure_outcome
from .pipeline_orchestrator import PipelineOrchestrator
from .complex_step_router import ComplexStepRouter
from .import_orchestrator import ImportOrchestrator
from .validation_orchestrator import ValidationOrchestrator
from .step_history_tracker import StepHistoryTracker
from ...steps.cache_step import CacheStep
from .policy_handlers import PolicyHandlers
from .dispatch_handler import DispatchHandler
from .result_handler import ResultHandler
from .telemetry_handler import TelemetryHandler
from .step_handler import StepHandler
from .agent_handler import AgentHandler
from .optimization_support import (
    OptimizationConfig,
    coerce_optimization_config,
    export_config as export_opt_config,
    get_config_manager as get_opt_config_manager,
    get_optimization_stats as get_opt_stats,
    get_performance_recommendations as get_opt_recommendations,
)
from .step_policies import (
    AgentResultUnpacker,
    AgentStepExecutor,
    CacheStepExecutor,
    ConditionalStepExecutor,
    DefaultAgentResultUnpacker,
    DefaultAgentStepExecutor,
    DefaultCacheStepExecutor,
    DefaultConditionalStepExecutor,
    DefaultDynamicRouterStepExecutor,
    DefaultHitlStepExecutor,
    DefaultLoopStepExecutor,
    DefaultParallelStepExecutor,
    DefaultPluginRedirector,
    DefaultSimpleStepExecutor,
    DefaultTimeoutRunner,
    DefaultValidatorInvoker,
    DynamicRouterStepExecutor,
    HitlStepExecutor,
    LoopStepExecutor,
    ParallelStepExecutor,
    PluginRedirector,
    SimpleStepExecutor,
    TimeoutRunner,
    ValidatorInvoker,
)
from .policy_primitives import PolicyRegistry
from .types import TContext_w_Scratch, ExecutionFrame
from .context_manager import ContextManager
from .context_update_manager import ContextUpdateManager
from .estimation import (
    UsageEstimator,
    HeuristicUsageEstimator,
    UsageEstimatorFactory,
    build_default_estimator_factory,
)
from .default_components import (
    DefaultAgentRunner,
    DefaultProcessorPipeline,
    DefaultValidatorRunner,
    DefaultPluginRunner,
    ThreadSafeMeter,
    InMemoryLRUBackend,
    DefaultTelemetry,
    OrjsonSerializer,
    Blake3Hasher,
    DefaultCacheKeyGenerator,
    _LRUCache,
)

# Re-export core protocols for compatibility with tests and external imports
from .executor_protocols import (
    ISerializer,
    IHasher,
    ICacheBackend,
    IUsageMeter,
)
# Protocols are defined in executor_protocols.py. They are not imported here
# to avoid unused-import lint warnings, as ExecutorCore uses structural typing
# and accepts concrete implementations via dependency injection.

# Module-level defaults for strictness to avoid per-instance configuration overhead
try:
    from ...infra.settings import get_settings as _get_settings_default

    _SETTINGS_DEFAULTS = _get_settings_default()
    _DEFAULT_STRICT_CONTEXT_ISOLATION: bool = bool(
        getattr(_SETTINGS_DEFAULTS, "strict_context_isolation", False)
    )
    _DEFAULT_STRICT_CONTEXT_MERGE: bool = bool(
        getattr(_SETTINGS_DEFAULTS, "strict_context_merge", False)
    )
except Exception:
    _DEFAULT_STRICT_CONTEXT_ISOLATION = False
    _DEFAULT_STRICT_CONTEXT_MERGE = False


# Backward-compatible error types used by other modules
class RetryableError(Exception):
    """Base class for errors that should trigger retries."""

    pass


class ValidationError(RetryableError):
    """Validation failures that can be retried."""

    pass


class PluginError(RetryableError):
    """Plugin failures that can be retried."""

    pass


class AgentError(RetryableError):
    """Agent execution errors that can be retried."""

    pass


class ContextIsolationError(Exception):
    """Raised when context isolation fails under strict settings."""

    pass


class ContextMergeError(Exception):
    """Raised when context merging fails under strict settings."""

    pass


# Compatibility class for tests
@dataclass
class _Frame:
    """Frame class for backward compatibility with tests."""

    step: Any
    data: Any
    context: Optional[Any] = None
    resources: Optional[Any] = None


# Type alias for step executor function signature
StepExecutor = Callable[
    [Step[Any, Any], Any, Optional[Any], Optional[Any], Optional[Any]],
    Awaitable[StepResult],
]


class ExecutorCore(Generic[TContext_w_Scratch]):
    """
    Policy-driven step executor with modular architecture.

    - Consistent step routing in execute()
    - Policy-based handlers for agents, loops, parallel, conditionals, routers, cache, and HITL
    - Proper isolation/merging of context across branches and retries
    - Centralized telemetry and usage metering
    """

    # Context variables moved to respective manager classes

    def __init__(
        self,
        agent_runner: Any = None,
        processor_pipeline: Any = None,
        validator_runner: Any = None,
        plugin_runner: Any = None,
        usage_meter: Any = None,
        quota_manager: Optional[QuotaManager] = None,
        cache_backend: Any = None,
        cache_key_generator: Any = None,
        telemetry: Any = None,
        enable_cache: bool = True,
        # Additional parameters for compatibility
        serializer: Any = None,
        hasher: Any = None,
        # UltraStepExecutor compatibility parameters
        cache_size: int = 1024,
        cache_ttl: int = 3600,
        concurrency_limit: int = 10,
        # Additional compatibility parameters
        optimization_config: Any = None,
        # Injected policies
        timeout_runner: Optional[TimeoutRunner] = None,
        unpacker: Optional[AgentResultUnpacker] = None,
        plugin_redirector: Optional[PluginRedirector] = None,
        validator_invoker: Optional[ValidatorInvoker] = None,
        simple_step_executor: Optional[SimpleStepExecutor] = None,
        agent_step_executor: Optional[AgentStepExecutor] = None,
        loop_step_executor: Optional[LoopStepExecutor] = None,
        parallel_step_executor: Optional[ParallelStepExecutor] = None,
        conditional_step_executor: Optional[ConditionalStepExecutor] = None,
        dynamic_router_step_executor: Optional[DynamicRouterStepExecutor] = None,
        hitl_step_executor: Optional[HitlStepExecutor] = None,
        cache_step_executor: Optional[CacheStepExecutor] = None,
        usage_estimator: Optional[UsageEstimator] = None,
        estimator_factory: Optional[UsageEstimatorFactory] = None,
        # Strict behavior toggles (robust defaults with optional enforcement)
        strict_context_isolation: bool = False,
        strict_context_merge: bool = False,
        enable_optimized_error_handling: bool = True,
        state_providers: Optional[Dict[str, StateProvider]] = None,
    ) -> None:
        # Validate parameters for compatibility
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if cache_ttl < 0:
            raise ValueError("cache_ttl must be non-negative")
        if concurrency_limit is not None and concurrency_limit <= 0:
            raise ValueError("concurrency_limit must be positive if specified")

        self.optimization_config: OptimizationConfig = coerce_optimization_config(
            optimization_config
        )
        config_issues = self.optimization_config.validate()
        if config_issues:
            warnings.warn(
                "OptimizationConfig validation issues: " + ", ".join(config_issues),
                RuntimeWarning,
                stacklevel=2,
            )

        effective_enable_optimized_error_handling = (
            bool(self.optimization_config.enable_optimized_error_handling)
            if optimization_config is not None
            else bool(enable_optimized_error_handling)
        )
        self.optimization_config.enable_optimized_error_handling = (
            effective_enable_optimized_error_handling
        )

        self._agent_runner = agent_runner or DefaultAgentRunner()
        self._processor_pipeline = processor_pipeline or DefaultProcessorPipeline()
        self._validator_runner = validator_runner or DefaultValidatorRunner()
        self._plugin_runner = plugin_runner or DefaultPluginRunner()
        self._usage_meter = usage_meter or ThreadSafeMeter()
        self._cache_manager = CacheManager(
            backend=cache_backend or InMemoryLRUBackend(max_size=cache_size, ttl_s=cache_ttl),
            key_generator=cache_key_generator,
            enable_cache=enable_cache,
        )

        # Backward compatibility properties for tests
        self._cache_backend = self._cache_manager.backend
        self._fallback_handler = FallbackHandler()
        self._telemetry = telemetry or DefaultTelemetry()
        self._enable_cache = enable_cache
        self.enable_optimized_error_handling = effective_enable_optimized_error_handling
        # Estimation selection: factory first, then direct estimator, then default
        self._estimator_factory: UsageEstimatorFactory = (
            estimator_factory or build_default_estimator_factory()
        )
        self._usage_estimator: UsageEstimator = usage_estimator or HeuristicUsageEstimator()
        self._step_history_tracker = StepHistoryTracker()
        self._quota_manager = quota_manager or QuotaManager()
        self._concurrency_limit = concurrency_limit

        # Backward compatibility properties for tests
        # Note: These are set after all managers are initialized

        # Store additional components for compatibility
        self._serializer = serializer or OrjsonSerializer()
        self._hasher = hasher or Blake3Hasher()
        self._cache_key_generator = cache_key_generator or DefaultCacheKeyGenerator(self._hasher)

        self._cache_locks: Dict[str, asyncio.Lock] = {}
        self._cache_locks_lock = asyncio.Lock()

        # Strict behavior settings (defaults can be overridden by global settings)
        self._strict_context_isolation = (
            bool(strict_context_isolation) or _DEFAULT_STRICT_CONTEXT_ISOLATION
        )
        self._strict_context_merge = bool(strict_context_merge) or _DEFAULT_STRICT_CONTEXT_MERGE
        self._hydration_manager = HydrationManager(state_providers)
        self._hydration_manager.set_telemetry(self._telemetry)
        self._background_task_manager = BackgroundTaskManager()
        self._context_update_manager = ContextUpdateManager()
        self._agent_orchestrator = AgentOrchestrator(plugin_runner=self._plugin_runner)
        self._conditional_orchestrator = ConditionalOrchestrator()
        self._hitl_orchestrator = HitlOrchestrator()
        self._loop_orchestrator = LoopOrchestrator()
        self._pipeline_orchestrator = PipelineOrchestrator()
        self._complex_step_router = ComplexStepRouter()
        self._validation_orchestrator = ValidationOrchestrator()

        # Backward compatibility properties for tests
        self._state_providers = self._hydration_manager._state_providers

        # Assign policies
        self.timeout_runner = timeout_runner or DefaultTimeoutRunner()
        self.unpacker = unpacker or DefaultAgentResultUnpacker()
        self.plugin_redirector = plugin_redirector or DefaultPluginRedirector(
            self._plugin_runner, self._agent_runner
        )
        self.validator_invoker = validator_invoker or DefaultValidatorInvoker(
            self._validator_runner
        )
        self.simple_step_executor = simple_step_executor or DefaultSimpleStepExecutor()
        self.agent_step_executor = agent_step_executor or DefaultAgentStepExecutor()
        self.loop_step_executor = loop_step_executor or DefaultLoopStepExecutor()
        self.parallel_step_executor = parallel_step_executor or DefaultParallelStepExecutor()
        self.conditional_step_executor = (
            conditional_step_executor or DefaultConditionalStepExecutor()
        )
        self.dynamic_router_step_executor = (
            dynamic_router_step_executor or DefaultDynamicRouterStepExecutor()
        )
        self.hitl_step_executor = hitl_step_executor or DefaultHitlStepExecutor()
        self.cache_step_executor = cache_step_executor or DefaultCacheStepExecutor()
        # ImportStep executor (policy-driven)
        from typing import Optional as _Optional
        from .step_policies import (
            DefaultImportStepExecutor as _DefaultImportStepExecutor,
            ImportStepExecutor as _ImportStepExecutor,
        )

        self.import_step_executor: _Optional[_ImportStepExecutor] = None
        try:
            self.import_step_executor = _DefaultImportStepExecutor()
        except Exception:  # pragma: no cover - defensive
            self.import_step_executor = None
        self._import_orchestrator = ImportOrchestrator(self.import_step_executor)

        # FSD-010: Initialize and populate the policy registry used for dispatch
        self.policy_registry = PolicyRegistry()

        # Policy handlers (delegated for composition-root slimming)
        self._policy_handlers = PolicyHandlers(self)
        self._policy_cache_step = self._policy_handlers.cache_step
        self._policy_import_step = self._policy_handlers.import_step
        self._policy_parallel_step = self._policy_handlers.parallel_step
        self._policy_loop_step = self._policy_handlers.loop_step
        self._policy_conditional_step = self._policy_handlers.conditional_step
        self._policy_dynamic_router_step = self._policy_handlers.dynamic_router_step
        self._policy_hitl_step = self._policy_handlers.hitl_step
        self._policy_default_step = self._policy_handlers.default_step

        # Register policies via delegated handler to keep ExecutorCore slim
        self._policy_handlers.register_all(self.policy_registry)

        # Dispatcher delegates to the shared policy registry
        self._dispatcher = ExecutionDispatcher(self.policy_registry)
        self._dispatch_handler = DispatchHandler(self)
        self._result_handler = ResultHandler(self)
        self._telemetry_handler = TelemetryHandler(self)
        self._step_handler = StepHandler(self)
        self._agent_handler = AgentHandler(self)

        # Initialize orchestrators that depend on executors registered above

    @property
    def cache(self) -> _LRUCache:
        return self._cache_manager.get_internal_cache()

    def clear_cache(self) -> None:
        self._cache_manager.clear_cache()

    def _cache_key(self, frame: Any) -> str:
        return self._cache_manager.generate_cache_key(
            frame.step, frame.data, frame.context, getattr(frame, "resources", None)
        )

    def _normalize_frame_context(self, frame: ExecutionFrame[Any]) -> None:
        """Strip mock contexts and retain hot-path accessors on the frame."""
        try:
            from unittest.mock import Mock as _Mock, MagicMock as _MagicMock

            try:
                from unittest.mock import AsyncMock as _AsyncMock

                _mock_types_ctx: tuple[type[Any], ...] = (_Mock, _MagicMock, _AsyncMock)
            except Exception:
                _mock_types_ctx = (_Mock, _MagicMock)
            ctx_attr = getattr(frame, "context", None)
            if isinstance(ctx_attr, _mock_types_ctx):
                setattr(frame, "context", None)
        except Exception:
            pass
        # Accessors kept for completeness; policies pull from frame directly
        _ = getattr(frame, "context", None)
        _ = getattr(frame, "resources", None)
        _ = getattr(frame, "limits", None)
        _ = getattr(frame, "on_chunk", None)

    async def _set_quota_and_hydrate(self, frame: ExecutionFrame[Any]) -> None:
        """Assign quota to the execution context and hydrate managed state."""
        self._set_current_quota(getattr(frame, "quota", None))
        await self._hydration_manager.hydrate_context(getattr(frame, "context", None))

    def _handle_missing_agent_exception(
        self, err: MissingAgentError, step: Any, *, called_with_frame: bool
    ) -> StepOutcome[StepResult] | StepResult:
        return self._result_handler.handle_missing_agent_exception(
            err, step, called_with_frame=called_with_frame
        )

    async def _persist_and_finalize(
        self,
        *,
        step: Any,
        result: StepResult,
        cache_key: Optional[str],
        called_with_frame: bool,
    ) -> StepOutcome[StepResult] | StepResult:
        return await self._result_handler.persist_and_finalize(
            step=step, result=result, cache_key=cache_key, called_with_frame=called_with_frame
        )

    def _handle_unexpected_exception(
        self,
        *,
        step: Any,
        frame: ExecutionFrame[Any],
        exc: Exception,
        called_with_frame: bool,
    ) -> StepOutcome[StepResult] | StepResult:
        return self._result_handler.handle_unexpected_exception(
            step=step, frame=frame, exc=exc, called_with_frame=called_with_frame
        )

    async def _maybe_use_cache(
        self, frame: ExecutionFrame[Any], *, called_with_frame: bool
    ) -> tuple[Optional[StepOutcome[StepResult] | StepResult], Optional[str]]:
        return await self._result_handler.maybe_use_cache(
            frame, called_with_frame=called_with_frame
        )

    def _log_step_start(self, step: Any, *, stream: bool, fallback_depth: int) -> None:
        self._telemetry_handler.log_step_start(step, stream=stream, fallback_depth=fallback_depth)

    async def _maybe_launch_background(
        self, frame: ExecutionFrame[Any], *, called_with_frame: bool
    ) -> Optional[StepOutcome[StepResult] | StepResult]:
        """Launch background step if applicable and return outcome."""
        try:
            bg_outcome = await self._background_task_manager.maybe_launch_background_step(
                core=self, frame=frame
            )
        except Exception:
            return None
        if bg_outcome is None:
            return None
        if called_with_frame:
            return bg_outcome
        return self._unwrap_outcome_to_step_result(bg_outcome, self._safe_step_name(frame.step))

    def _get_current_quota(self) -> Optional[Quota]:
        """Best-effort getter for the current quota using the manager first."""
        try:
            quota = self._quota_manager.get_current_quota()
            if quota is not None:
                return quota
        except Exception:
            pass
        return None

    def _set_current_quota(self, quota: Optional[Quota]) -> Optional[object]:
        """Best-effort setter for the current quota (returns token when available)."""
        try:
            return self._quota_manager.set_current_quota(quota)
        except Exception:
            return None

    def _reset_current_quota(self, token: Optional[object]) -> None:
        """Best-effort reset for quota context tokens."""
        try:
            if token is not None and hasattr(token, "old_value"):
                self._quota_manager.set_current_quota(token.old_value)
                return
        except Exception:
            pass

    def _hash_obj(self, obj: Any) -> str:
        if obj is None:
            return "None"
        elif isinstance(obj, bytes):
            return self._hasher.digest(obj)
        elif isinstance(obj, str):
            return self._hasher.digest(obj.encode("utf-8"))
        else:
            try:
                serialized = self._serializer.serialize(obj)
                return self._hasher.digest(serialized)
            except Exception:
                return self._hasher.digest(str(obj).encode("utf-8"))

    def _isolate_context(
        self, context: Optional[TContext_w_Scratch]
    ) -> Optional[TContext_w_Scratch]:
        if context is None:
            return None
        if self._strict_context_isolation:
            # Delegate strict isolation
            isolated = ContextManager.isolate_strict(cast(Optional[BaseModel], context))
        else:
            # Non-strict isolation
            isolated = ContextManager.isolate(cast(Optional[BaseModel], context))
        return cast(Optional[TContext_w_Scratch], isolated)

    def _merge_context_updates(
        self,
        main_context: Optional[TContext_w_Scratch],
        branch_context: Optional[TContext_w_Scratch],
    ) -> Optional[TContext_w_Scratch]:
        if main_context is None and branch_context is None:
            return None
        elif main_context is None:
            return branch_context
        elif branch_context is None:
            return main_context

        # Delegate to ContextManager with strict toggle
        if self._strict_context_merge:
            merged = ContextManager.merge_strict(
                cast(Optional[BaseModel], main_context),
                cast(Optional[BaseModel], branch_context),
            )
        else:
            merged = ContextManager.merge(
                cast(Optional[BaseModel], main_context),
                cast(Optional[BaseModel], branch_context),
            )
        return cast(Optional[TContext_w_Scratch], merged)

    def _accumulate_loop_context(
        self,
        current_context: Optional[TContext_w_Scratch],
        iteration_context: Optional[TContext_w_Scratch],
    ) -> Optional[TContext_w_Scratch]:
        if current_context is None:
            return iteration_context
        elif iteration_context is None:
            return current_context
        merged_context = self._merge_context_updates(current_context, iteration_context)
        return merged_context

    def _update_context_state(self, context: Optional[TContext_w_Scratch], state: str) -> None:
        if context is None:
            return
        try:
            if hasattr(context, "scratchpad"):
                try:
                    scratchpad = getattr(context, "scratchpad")
                    if isinstance(scratchpad, dict):
                        scratchpad["state"] = state
                    else:
                        setattr(scratchpad, "state", state)
                except Exception:
                    pass
        except Exception:
            pass

    def _is_complex_step(self, step: Any) -> bool:
        # 1) Explicit object-oriented hook/property
        try:
            if hasattr(step, "is_complex"):
                prop = getattr(step, "is_complex")
                if callable(prop):
                    try:
                        if bool(prop()):
                            return True
                    except Exception:
                        pass
                else:
                    if bool(prop):
                        return True
        except Exception:
            pass

        # 2) Known complex step types
        if isinstance(
            step,
            (
                ParallelStep,
                LoopStep,
                ConditionalStep,
                DynamicParallelRouterStep,
                HumanInTheLoopStep,
                CacheStep,
            ),
        ):
            return True

        # 3) Backward-compat: name-based signal
        try:
            if getattr(step, "name", None) == "cache":
                return True
        except Exception:
            pass

        # 4) Backward-compat: validation/plugin/fallback configuration implies complexity
        try:
            plugins = getattr(step, "plugins", None)
            if isinstance(plugins, (list, tuple)):
                if len(plugins) > 0:
                    return True
            elif plugins:
                return True
        except Exception:
            pass
        # 5) Backward-compat: validation-step metadata
        try:
            if hasattr(step, "meta") and isinstance(step.meta, dict):
                if step.meta.get("is_validation_step"):
                    return True
        except Exception:
            pass
        return False

    # ------------------------
    # Outcome normalization
    # ------------------------
    def _unwrap_outcome_to_step_result(
        self, outcome: StepOutcome[StepResult] | StepResult, step_name: str
    ) -> StepResult:
        return self._result_handler.unwrap_outcome_to_step_result(outcome, step_name)

    async def _dispatch_frame(
        self, frame: ExecutionFrame[Any], *, called_with_frame: bool
    ) -> StepOutcome[StepResult] | StepResult:
        return await self._dispatch_handler.dispatch(frame, called_with_frame=called_with_frame)

    async def _execute_complex_step(
        self,
        *,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
        _fallback_depth: int = 0,
    ) -> StepResult:
        try:
            fb_depth = int(_fallback_depth)
        except Exception:
            fb_depth = 0
        return await self._complex_step_router.route(
            core=self,
            step=step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            stream=stream,
            on_chunk=on_chunk,
            context_setter=context_setter,
            fallback_depth=fb_depth,
        )

    async def wait_for_background_tasks(self, timeout: float = 5.0) -> None:
        """Wait for all background tasks to complete with a timeout."""
        await self._background_task_manager.wait_for_completion(timeout)

    async def execute(
        self,
        frame_or_step: Any | None = None,
        data: Any | None = None,
        **kwargs: Any,
    ) -> StepOutcome[StepResult] | StepResult:
        # Support both ExecutionFrame and legacy signature (step, data, ...)
        from typing import cast

        called_with_frame = isinstance(frame_or_step, ExecutionFrame)
        if called_with_frame:
            frame = cast(ExecutionFrame[Any], frame_or_step)
        else:
            # Accept duck-typed steps for backward compatibility
            step_obj = frame_or_step if frame_or_step is not None else kwargs.get("step")
            if step_obj is None:
                raise ValueError("ExecutorCore.execute requires a Step or ExecutionFrame")
            payload = data if data is not None else kwargs.get("data")
            fb_depth_raw = kwargs.get("_fallback_depth", 0)
            try:
                fb_depth_norm = int(fb_depth_raw)
            except Exception:
                fb_depth_norm = 0
            frame = self._make_execution_frame(
                cast(Step[Any, Any], step_obj),
                payload,
                kwargs.get("context"),
                kwargs.get("resources"),
                kwargs.get("limits"),
                kwargs.get("context_setter"),
                stream=kwargs.get("stream", False),
                on_chunk=kwargs.get("on_chunk"),
                fallback_depth=fb_depth_norm,
                quota=kwargs.get("quota"),
                result=kwargs.get("result"),
            )

        await self._set_quota_and_hydrate(frame)

        step = frame.step

        # Check for background execution mode before dispatch (delegated to manager)
        bg_outcome = await self._maybe_launch_background(frame, called_with_frame=called_with_frame)
        if bg_outcome is not None:
            return bg_outcome

        self._normalize_frame_context(frame)
        stream = getattr(frame, "stream", False)
        _fallback_depth = getattr(frame, "_fallback_depth", 0)

        self._log_step_start(step, stream=stream, fallback_depth=_fallback_depth)

        # FSD-009: Remove reactive post-step usage checks from non-parallel codepaths.
        # Preemptive quota reservations are enforced in policies; no reactive governor path remains.

        cached_outcome, cache_key = await self._maybe_use_cache(
            frame, called_with_frame=called_with_frame
        )
        if cached_outcome is not None:
            return cached_outcome

        try:
            result_outcome = await self._dispatch_frame(frame, called_with_frame=called_with_frame)
        except MissingAgentError as e:
            handled = self._handle_missing_agent_exception(
                e, step, called_with_frame=called_with_frame
            )
            if handled is not None:
                return handled
            raise
        except (UsageLimitExceededError, MockDetectionError, InfiniteFallbackError):
            raise
        except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
            raise
        except Exception as exc:
            return self._handle_unexpected_exception(
                step=step, frame=frame, exc=exc, called_with_frame=called_with_frame
            )

        if isinstance(result_outcome, StepOutcome):
            return result_outcome
        result = result_outcome

        return await self._persist_and_finalize(
            step=step, result=result, cache_key=cache_key, called_with_frame=called_with_frame
        )

    def _log_execution_error(self, step_name: str, exc: Exception) -> None:
        self._telemetry_handler.log_execution_error(step_name, exc)

    def _build_failure_outcome(
        self,
        *,
        step: Any,
        frame: ExecutionFrame[Any],
        exc: Exception,
        called_with_frame: bool,
    ) -> Failure[StepResult]:
        return build_failure_outcome(
            step=step,
            frame=frame,
            exc=exc,
            called_with_frame=called_with_frame,
            safe_step_name=self._safe_step_name,
        )

    def _make_execution_frame(
        self,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        *,
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        fallback_depth: int = 0,
        quota: Optional[Quota] = None,
        result: Optional[StepResult] = None,
    ) -> ExecutionFrame[Any]:
        """Create an ExecutionFrame with the current quota context."""
        return ExecutionFrame(
            step=step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            quota=quota if quota is not None else self._quota_manager.get_current_quota(),
            stream=stream,
            on_chunk=on_chunk,
            context_setter=context_setter or (lambda _res, _ctx: None),
            result=result,
            _fallback_depth=fallback_depth,
        )

    # Compatibility shim for existing call sites/tests
    async def execute_step(
        self,
        step: Any,
        data: Any,
        context: Optional[TContext_w_Scratch] = None,
        resources: Optional[Any] = None,
        limits: Optional[UsageLimits] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
        result: Optional[StepResult] = None,
        _fallback_depth: int = 0,
        usage_limits: Optional[UsageLimits] = None,
    ) -> StepResult:
        if usage_limits is not None and limits is None:
            limits = usage_limits
        try:
            outcome = await self.execute(
                step,
                data,
                context=context,
                resources=resources,
                limits=limits,
                stream=stream,
                on_chunk=on_chunk,
                context_setter=context_setter,
                result=result,
                _fallback_depth=_fallback_depth,
            )
        except InfiniteFallbackError:
            # Re-raise to satisfy unified error handling for streaming and control-flow
            raise
        return self._unwrap_outcome_to_step_result(outcome, self._safe_step_name(step))

    async def _execute_simple_step(
        self,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]],
        cache_key: Optional[str],
        _fallback_depth: int | None = 0,
        _from_policy: bool = False,
    ) -> StepResult:
        # Normalize fallback depth to an int to keep comparisons safe in policies
        fb_depth: int
        try:
            fb_depth = int(_fallback_depth) if _fallback_depth is not None else 0
        except Exception:
            fb_depth = 0
        outcome = await self.simple_step_executor.execute(
            self,
            step,
            data,
            context,
            resources,
            limits,
            stream,
            on_chunk,
            cache_key,
            fb_depth,
        )
        return self._unwrap_outcome_to_step_result(outcome, self._safe_step_name(step))

    async def _handle_parallel_step(
        self,
        step: Any | None = None,
        data: Any | None = None,
        context: Any | None = None,
        resources: Any | None = None,
        limits: Any | None = None,
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
        *,
        parallel_step: Any | None = None,
        step_executor: Optional[Callable[..., Awaitable[StepResult]]] = None,
    ) -> StepResult:
        ps = parallel_step if parallel_step is not None else step
        return await self._step_handler.parallel_step(
            ps, data, context, resources, limits, context_setter, step_executor
        )

    async def _execute_pipeline(
        self,
        pipeline: Any,
        data: Any,
        context: Any,
        resources: Any,
        limits: Any,
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
    ) -> PipelineResult[Any]:
        return await self._execute_pipeline_via_policies(
            pipeline, data, context, resources, limits, context_setter
        )

    async def _execute_pipeline_via_policies(
        self,
        pipeline: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[Any],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
    ) -> PipelineResult[Any]:
        return await self._step_handler.pipeline(
            pipeline, data, context, resources, limits, context_setter
        )

    async def _handle_loop_step(
        self,
        loop_step: Any,
        data: Any,
        context: Any,
        resources: Any,
        limits: Any,
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        _fallback_depth: int = 0,
    ) -> StepResult:
        return await self._step_handler.loop_step(
            loop_step, data, context, resources, limits, context_setter, _fallback_depth
        )

    async def _handle_dynamic_router(
        self,
        step: Any | None = None,
        data: Any | None = None,
        context: Any | None = None,
        resources: Any | None = None,
        limits: Any | None = None,
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
        router_step: Any | None = None,
        step_executor: Optional[Callable[..., Awaitable[StepResult]]] = None,
    ) -> StepResult:
        rs = router_step if router_step is not None else step
        return await self._step_handler.dynamic_router_wrapper(
            step,
            data,
            context,
            resources,
            limits,
            context_setter,
            rs,
            step_executor,
        )

    async def _handle_hitl_step(
        self,
        step: HumanInTheLoopStep,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        cache_key: Optional[str] = None,
        _fallback_depth: int = 0,
        **kwargs: Any,
    ) -> StepResult:
        return await self._step_handler.hitl_step(
            step,
            data,
            context,
            resources,
            limits,
            context_setter,
            stream,
            on_chunk,
            cache_key,
            _fallback_depth,
        )

    # Legacy compatibility wrappers expected by tests
    async def _execute_loop(
        self,
        loop_step: Any,
        data: Any,
        context: Any,
        resources: Any,
        limits: Any,
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        _fallback_depth: int = 0,
    ) -> StepResult:
        return await self._loop_orchestrator.execute(
            core=self,
            loop_step=loop_step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            context_setter=context_setter,
            fallback_depth=_fallback_depth,
        )

    async def _handle_cache_step(
        self,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
        *,
        step_executor: Optional[Callable[..., Awaitable[StepResult]]] = None,
        **kwargs: Any,
    ) -> StepResult:
        return await self._step_handler.cache_step(
            step, data, context, resources, limits, context_setter, step_executor
        )

    async def _handle_conditional_step(
        self,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        _fallback_depth: int = 0,
        **kwargs: Any,
    ) -> StepResult:
        return await self._step_handler.conditional_step(
            step, data, context, resources, limits, context_setter, _fallback_depth
        )

    async def _handle_dynamic_router_step(
        self,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
        **kwargs: Any,
    ) -> StepResult:
        return await self._step_handler.dynamic_router_step(
            step, data, context, resources, limits, context_setter
        )

    def _default_set_final_context(
        self, result: PipelineResult[Any], context: Optional[Any]
    ) -> None:
        pass

    def _safe_step_name(self, step: Any) -> str:
        try:
            if hasattr(step, "name"):
                name = step.name
                if hasattr(name, "_mock_name"):
                    if hasattr(name, "_mock_return_value") and name._mock_return_value:
                        return str(name._mock_return_value)
                    elif hasattr(name, "_mock_name") and name._mock_name:
                        return str(name._mock_name)
                    else:
                        return "mock_step"
                else:
                    return str(name)
            else:
                return "unknown_step"
        except Exception:
            return "unknown_step"

    def _format_feedback(
        self, feedback: Optional[str], default_message: str = "Agent execution failed"
    ) -> str:
        if feedback is None or str(feedback).strip() == "":
            return default_message
        msg = str(feedback)
        low = msg.lower()
        if ("error" not in low) and ("exception" not in low):
            base = default_message or "Agent exception"
            if ("error" not in base.lower()) and ("exception" not in base.lower()):
                base = "Agent exception"
            return f"{base}: {msg}"
        return msg

    # --- Orchestrated Simple Agent Step (centralized control flow) ---

    # --- Orchestrated Simple Agent Step (centralized control flow) ---
    async def _execute_agent_with_orchestration(
        self,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]],
        cache_key: Optional[str],
        _fallback_depth: int,
    ) -> StepOutcome[StepResult]:
        """Compatibility wrapper: agent orchestration now lives in AgentOrchestrator."""
        return await self._agent_handler.execute(
            step=step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            stream=stream,
            on_chunk=on_chunk,
            cache_key=cache_key,
            fallback_depth=_fallback_depth,
        )

    def get_optimization_stats(self) -> dict[str, Any]:
        return get_opt_stats(self.optimization_config)

    def get_config_manager(self) -> Any:
        return get_opt_config_manager(self.optimization_config)

    def get_performance_recommendations(self) -> list[dict[str, Any]]:
        return get_opt_recommendations()

    def export_config(self, format_type: str = "dict") -> dict[str, Any]:
        return export_opt_config(self.optimization_config, format_type)


# Backward compatibility: lightweight usage tracker used by some tests
@dataclass
class _UsageTracker:
    total_cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _lock: asyncio.Lock = asyncio.Lock()

    async def add(self, cost_usd: float, tokens: int) -> None:
        async with self._lock:
            self.total_cost_usd += float(cost_usd)
            self.prompt_tokens += int(tokens)

    async def guard(self, limits: UsageLimits) -> None:
        async with self._lock:
            if (
                limits.total_cost_usd_limit is not None
                and self.total_cost_usd > limits.total_cost_usd_limit
            ):
                raise UsageLimitExceededError("Cost limit exceeded")
            total_tokens = self.prompt_tokens + self.completion_tokens
            if limits.total_tokens_limit is not None and total_tokens > limits.total_tokens_limit:
                raise UsageLimitExceededError("Token limit exceeded")

    async def snapshot(self) -> tuple[float, int, int]:
        async with self._lock:
            return self.total_cost_usd, self.prompt_tokens, self.completion_tokens

    async def get_current_totals(self) -> tuple[float, int]:
        async with self._lock:
            total_tokens = self.prompt_tokens + self.completion_tokens
            return self.total_cost_usd, total_tokens


class OptimizedExecutorCore(ExecutorCore[Any]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "OptimizedExecutorCore is deprecated; use ExecutorCore with OptimizationConfig.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = [
    "ExecutorCore",
    "PluginError",
    "StepExecutor",
    "_UsageTracker",
    "OptimizedExecutorCore",
    "OptimizationConfig",
    # Re-exports for compatibility
    "ISerializer",
    "IHasher",
    "ICacheBackend",
    "IUsageMeter",
    "OrjsonSerializer",
    "Blake3Hasher",
    "InMemoryLRUBackend",
    "ThreadSafeMeter",
    "DefaultAgentRunner",
    "DefaultProcessorPipeline",
    "DefaultValidatorRunner",
    "DefaultPluginRunner",
    "DefaultTelemetry",
]
