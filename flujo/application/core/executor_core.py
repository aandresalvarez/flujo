from __future__ import annotations
import asyncio
import warnings
from typing import Any, Awaitable, Callable, Dict, Generic, Optional, cast, TYPE_CHECKING

from ...domain.interfaces import StateProvider
from ...domain.models import PipelineResult, Quota, StepOutcome, StepResult, UsageLimits
from ...exceptions import (
    MissingAgentError,
)
from ...infra.settings import get_settings
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
from .policy_handlers import PolicyHandlers
from .dispatch_handler import DispatchHandler
from .result_handler import ResultHandler
from .telemetry_handler import TelemetryHandler
from .step_handler import StepHandler
from .agent_handler import AgentHandler
from .optimization_support import (
    OptimizationConfig,
    OptimizedExecutorCore,
    coerce_optimization_config,
    export_config as export_opt_config,
    get_config_manager as get_opt_config_manager,
    get_optimization_stats as get_opt_stats,
    get_performance_recommendations as get_opt_recommendations,
)
from .executor_helpers import (
    _UsageTracker,
    format_feedback,
    safe_step_name,
    normalize_frame_context,
    set_quota_and_hydrate,
    get_current_quota,
    set_current_quota,
    reset_current_quota,
    hash_obj,
    isolate_context,
    merge_context_updates,
    accumulate_loop_context,
    update_context_state,
    is_complex_step,
    log_execution_error,
    make_execution_frame,
    execute_simple_step,
    execute_step_compat,
    build_failure,
    handle_missing_agent_exception,
    persist_and_finalize,
    handle_unexpected_exception,
    maybe_use_cache,
    execute_entrypoint,
    PluginError,
    StepExecutor,
)
from .executor_wrappers import (
    handle_parallel_step,
    execute_pipeline,
    execute_pipeline_via_policies,
    handle_loop_step,
    handle_dynamic_router,
    handle_hitl_step,
    execute_loop,
    handle_cache_step,
    handle_conditional_step,
    handle_dynamic_router_step,
    default_set_final_context,
    execute_agent_with_orchestration,
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
from .policy_registry import PolicyRegistry, create_default_registry
from .types import TContext_w_Scratch, ExecutionFrame
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
    DefaultTelemetry,
)
from .executor_protocols import (
    IAgentRunner,
    IProcessorPipeline,
    IValidatorRunner,
    IPluginRunner,
    IUsageMeter,
    ITelemetry,
)
from .default_cache_components import (
    ThreadSafeMeter,
    InMemoryLRUBackend,
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

if TYPE_CHECKING:
    from .state_manager import StateManager
    from ...type_definitions.common import JSONObject


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
        agent_runner: IAgentRunner | None = None,
        processor_pipeline: IProcessorPipeline | None = None,
        validator_runner: IValidatorRunner | None = None,
        plugin_runner: IPluginRunner | None = None,
        usage_meter: IUsageMeter | None = None,
        quota_manager: Optional[QuotaManager] = None,
        cache_backend: Any = None,
        cache_key_generator: Any = None,
        telemetry: ITelemetry | None = None,
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
        policy_registry: Optional[PolicyRegistry] = None,
        # Strict behavior toggles (robust defaults with optional enforcement)
        strict_context_isolation: bool = False,
        strict_context_merge: bool = False,
        enable_optimized_error_handling: bool = True,
        state_providers: Optional[Dict[str, StateProvider[Any]]] = None,
        state_manager: Optional["StateManager[Any]"] = None,
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
        self.state_manager = state_manager
        self._context_update_manager = ContextUpdateManager()
        self._agent_orchestrator = AgentOrchestrator(plugin_runner=self._plugin_runner)
        self._conditional_orchestrator = ConditionalOrchestrator()
        self._hitl_orchestrator = HitlOrchestrator()
        self._loop_orchestrator = LoopOrchestrator()
        self._pipeline_orchestrator = PipelineOrchestrator()
        self._complex_step_router = ComplexStepRouter()
        self._validation_orchestrator = ValidationOrchestrator()

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
        self.policy_registry = policy_registry or create_default_registry(self)

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
        self._dispatcher = ExecutionDispatcher(self.policy_registry, core=self)
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

    _normalize_frame_context = staticmethod(normalize_frame_context)

    async def _set_quota_and_hydrate(self, frame: ExecutionFrame[Any]) -> None:
        """Assign quota to the execution context and hydrate managed state."""
        await set_quota_and_hydrate(frame, self._quota_manager, self._hydration_manager)

    def _handle_missing_agent_exception(
        self, err: MissingAgentError, step: Any, *, called_with_frame: bool
    ) -> StepOutcome[StepResult] | StepResult:
        return handle_missing_agent_exception(self, err, step, called_with_frame=called_with_frame)

    async def _persist_and_finalize(
        self,
        *,
        step: Any,
        result: StepResult,
        cache_key: Optional[str],
        called_with_frame: bool,
    ) -> StepOutcome[StepResult] | StepResult:
        return await persist_and_finalize(
            self, step=step, result=result, cache_key=cache_key, called_with_frame=called_with_frame
        )

    def _handle_unexpected_exception(
        self,
        *,
        step: Any,
        frame: ExecutionFrame[Any],
        exc: Exception,
        called_with_frame: bool,
    ) -> StepOutcome[StepResult] | StepResult:
        return handle_unexpected_exception(
            self, step=step, frame=frame, exc=exc, called_with_frame=called_with_frame
        )

    async def _maybe_use_cache(
        self, frame: ExecutionFrame[Any], *, called_with_frame: bool
    ) -> tuple[Optional[StepOutcome[StepResult] | StepResult], Optional[str]]:
        return await maybe_use_cache(self, frame, called_with_frame=called_with_frame)

    def _get_current_quota(self) -> Optional[Quota]:
        """Best-effort getter for the current quota using the manager first."""
        return get_current_quota(self._quota_manager)

    def _set_current_quota(self, quota: Optional[Quota]) -> Optional[object]:
        """Best-effort setter for the current quota (returns token when available)."""
        return set_current_quota(self._quota_manager, quota)

    def _reset_current_quota(self, token: Optional[object]) -> None:
        """Best-effort reset for quota context tokens."""
        reset_current_quota(self._quota_manager, token)

    def _get_background_quota(self, parent_quota: Optional[Quota] = None) -> Optional[Quota]:
        """Compute quota for background tasks with parent-first split."""
        settings = get_settings()
        bg_settings = getattr(settings, "background_tasks", None)
        if bg_settings is None or not bool(getattr(bg_settings, "enable_quota", False)):
            return parent_quota

        if parent_quota is not None:
            try:
                return parent_quota.split(1)[0]
            except Exception:
                try:
                    from ...infra import telemetry as _telemetry

                    _telemetry.logfire.warning(
                        "Cannot split parent quota for background task; quota disabled for task"
                    )
                except Exception:
                    pass
                return None

        try:
            return Quota(
                float(getattr(bg_settings, "max_cost_per_task", 0.0)),
                int(getattr(bg_settings, "max_tokens_per_task", 0)),
            )
        except Exception:
            return None

    async def _register_background_task(
        self,
        *,
        task_id: str,
        bg_run_id: str,
        parent_run_id: Optional[str],
        step_name: str,
        data: Any,
        context: Optional[TContext_w_Scratch],
        metadata: Optional["JSONObject"] = None,
    ) -> None:
        """Persist initial state for a background task."""
        if self.state_manager is None:
            return

        meta = dict(metadata or {})
        meta.setdefault("is_background_task", True)
        meta.setdefault("task_id", task_id)
        meta.setdefault("parent_run_id", parent_run_id)
        meta.setdefault("step_name", step_name)
        meta.setdefault("input_data", data)

        await self.state_manager.persist_workflow_state(
            run_id=bg_run_id,
            context=context,
            current_step_index=0,
            last_step_output=None,
            status="running",
            metadata=meta,
        )

    async def _mark_background_task_completed(
        self,
        *,
        task_id: str,
        context: Optional[TContext_w_Scratch],
        metadata: Optional["JSONObject"] = None,
    ) -> None:
        """Mark a background task as completed."""
        if self.state_manager is None:
            return

        run_id = getattr(context, "run_id", None) if context is not None else None
        if run_id is None:
            return

        meta = dict(metadata or {})
        meta.setdefault("task_id", task_id)
        meta.setdefault("is_background_task", True)

        await self.state_manager.persist_workflow_state(
            run_id=run_id,
            context=context,
            current_step_index=1,
            last_step_output=None,
            status="completed",
            metadata=meta,
        )
        try:
            from ...infra import telemetry as _telemetry

            _telemetry.logfire.info(f"Background task '{task_id}' completed successfully")
        except Exception:
            pass

    async def _mark_background_task_failed(
        self,
        *,
        task_id: str,
        context: Optional[TContext_w_Scratch],
        error: Exception,
        metadata: Optional["JSONObject"] = None,
    ) -> None:
        """Mark a background task as failed."""
        if self.state_manager is None:
            return

        run_id = getattr(context, "run_id", None) if context is not None else None
        if run_id is None:
            return

        meta = dict(metadata or {})
        meta.setdefault("task_id", task_id)
        meta.setdefault("is_background_task", True)
        meta["background_error"] = meta.get("background_error") or str(error)

        if context is not None and hasattr(context, "scratchpad"):
            try:
                context.scratchpad["background_error"] = str(error)
            except Exception:
                pass

        await self.state_manager.persist_workflow_state(
            run_id=run_id,
            context=context,
            current_step_index=0,
            last_step_output=None,
            status="failed",
            metadata=meta,
        )
        try:
            from ...infra import telemetry as _telemetry

            _telemetry.logfire.error(
                f"Background task '{task_id}' failed", extra={"error": str(error)}
            )
        except Exception:
            pass

    async def _mark_background_task_paused(
        self,
        *,
        task_id: str,
        context: Optional[TContext_w_Scratch],
        error: Exception,
        metadata: Optional["JSONObject"] = None,
    ) -> None:
        """Mark a background task as paused (control-flow signal)."""
        if self.state_manager is None:
            return

        run_id = getattr(context, "run_id", None) if context is not None else None
        if run_id is None:
            return

        meta = dict(metadata or {})
        meta.setdefault("task_id", task_id)
        meta.setdefault("is_background_task", True)
        meta["background_error"] = meta.get("background_error") or str(error)

        await self.state_manager.persist_workflow_state(
            run_id=run_id,
            context=context,
            current_step_index=0,
            last_step_output=None,
            status="paused",
            metadata=meta,
        )
        try:
            from ...infra import telemetry as _telemetry

            _telemetry.logfire.info(
                f"Background task '{task_id}' paused", extra={"reason": str(error)}
            )
        except Exception:
            pass

    def _hash_obj(self, obj: Any) -> str:
        return hash_obj(obj, self._serializer, self._hasher)

    def _isolate_context(
        self, context: Optional[TContext_w_Scratch]
    ) -> Optional[TContext_w_Scratch]:
        return cast(
            Optional[TContext_w_Scratch],
            isolate_context(context, strict_context_isolation=bool(self._strict_context_isolation)),
        )

    def _merge_context_updates(
        self,
        main_context: Optional[TContext_w_Scratch],
        branch_context: Optional[TContext_w_Scratch],
    ) -> Optional[TContext_w_Scratch]:
        return cast(
            Optional[TContext_w_Scratch],
            merge_context_updates(
                main_context,
                branch_context,
                strict_context_merge=bool(self._strict_context_merge),
            ),
        )

    def _accumulate_loop_context(
        self,
        current_context: Optional[TContext_w_Scratch],
        iteration_context: Optional[TContext_w_Scratch],
    ) -> Optional[TContext_w_Scratch]:
        return cast(
            Optional[TContext_w_Scratch],
            accumulate_loop_context(
                current_context,
                iteration_context,
                strict_context_merge=bool(self._strict_context_merge),
            ),
        )

    _update_context_state = staticmethod(update_context_state)
    _is_complex_step = staticmethod(is_complex_step)

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
        frame_or_step: ExecutionFrame[Any] | Any | None = None,
        data: Any | None = None,
        **kwargs: Any,
    ) -> StepOutcome[StepResult] | StepResult:
        """Public entrypoint that delegates to the shared execution flow."""
        return await execute_entrypoint(self, frame_or_step, data, **kwargs)

    def _log_execution_error(self, step_name: str, exc: Exception) -> None:
        log_execution_error(self, step_name, exc)

    _build_failure_outcome = build_failure

    _make_execution_frame = make_execution_frame

    # Compatibility shim for existing call sites/tests
    execute_step = execute_step_compat

    _execute_simple_step = execute_simple_step

    # Preserve failure builder compatibility
    _failure_builder = build_failure_outcome

    _handle_parallel_step = handle_parallel_step
    _execute_pipeline = execute_pipeline
    _execute_pipeline_via_policies = execute_pipeline_via_policies
    _handle_loop_step = handle_loop_step
    _handle_dynamic_router = handle_dynamic_router
    _handle_hitl_step = handle_hitl_step
    _execute_loop = execute_loop
    _handle_cache_step = handle_cache_step
    _handle_conditional_step = handle_conditional_step
    _handle_dynamic_router_step = handle_dynamic_router_step
    _default_set_final_context = staticmethod(default_set_final_context)

    _safe_step_name = staticmethod(safe_step_name)
    _format_feedback = staticmethod(format_feedback)

    _execute_agent_with_orchestration = execute_agent_with_orchestration

    def get_optimization_stats(self) -> dict[str, Any]:
        return get_opt_stats(self.optimization_config)

    def get_config_manager(self) -> Any:
        return get_opt_config_manager(self.optimization_config)

    def get_performance_recommendations(self) -> list[dict[str, Any]]:
        return get_opt_recommendations()

    def export_config(self, format_type: str = "dict") -> dict[str, Any]:
        return export_opt_config(self.optimization_config, format_type)


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
    "DefaultCacheKeyGenerator",
]
