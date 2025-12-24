from __future__ import annotations
import asyncio

from collections.abc import Awaitable, Callable
from typing import Generic, TYPE_CHECKING, TypeVar

from ...domain.interfaces import StateProvider
from ...domain.sandbox import SandboxProtocol
from ...domain.memory import VectorStoreProtocol
from ...domain.models import BaseModel as DomainBaseModel
from ...domain.models import PipelineResult, Quota, StepOutcome, StepResult, UsageLimits
from ...exceptions import (
    MissingAgentError,
)
from ...utils.async_bridge import run_sync
from .quota_manager import QuotaManager
from .execution_dispatcher import ExecutionDispatcher
from .failure_builder import build_failure_outcome
from .policy_handlers import PolicyHandlers
from .dispatch_handler import DispatchHandler
from .result_handler import ResultHandler
from .telemetry_handler import TelemetryHandler
from .step_handler import StepHandler
from .agent_handler import AgentHandler
from .shadow_evaluator import ShadowEvaluator

from .runtime_builder import ExecutorCoreDeps, FlujoRuntimeBuilder
from .execution.executor_helpers import (
    PluginError,
    StepExecutor,
    _CACHE_OVERRIDE,
    _UsageTracker,
    accumulate_loop_context,
    build_failure,
    execute_entrypoint,
    format_feedback,
    get_current_quota,
    handle_missing_agent_exception,
    handle_unexpected_exception,
    hash_obj,
    is_complex_step,
    isolate_context,
    log_execution_error,
    make_execution_frame,
    maybe_use_cache,
    merge_context_updates,
    normalize_frame_context,
    persist_and_finalize,
    reset_current_quota,
    safe_step_name,
    set_current_quota,
    set_quota_and_hydrate,
    update_context_state,
)
from .step_policies import (
    AgentResultUnpacker,
    AgentStepExecutor,
    CacheStepExecutor,
    ConditionalStepExecutor,
    DynamicRouterStepExecutor,
    HitlStepExecutor,
    ImportStepExecutor,
    LoopStepExecutor,
    ParallelStepExecutor,
    PluginRedirector,
    SimpleStepExecutor,
    TimeoutRunner,
    ValidatorInvoker,
)
from .policy_registry import PolicyRegistry, create_default_registry
from .types import TContext, ExecutionFrame
from .estimation import (
    UsageEstimator,
    UsageEstimatorFactory,
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
from .runtime.default_cache_components import (
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
    from .agent_orchestrator import AgentOrchestrator
    from .conditional_orchestrator import ConditionalOrchestrator
    from .hitl_orchestrator import HitlOrchestrator
    from .import_orchestrator import ImportOrchestrator
    from .loop_orchestrator import LoopOrchestrator
    from .pipeline_orchestrator import PipelineOrchestrator
    from .validation_orchestrator import ValidationOrchestrator
    from .policy_registry import PolicyCallable
    from ...type_definitions.common import JSONObject


TFrameContext = TypeVar("TFrameContext", bound=DomainBaseModel)


class ExecutorCore(Generic[TContext]):
    """
    Policy-driven step executor with modular architecture.

    - Consistent step routing in execute()
    - Policy-based handlers for agents, loops, parallel, conditionals, routers, cache, and HITL
    - Proper isolation/merging of context across branches and retries
    - Centralized telemetry and usage metering
    """

    timeout_runner: TimeoutRunner
    unpacker: AgentResultUnpacker
    plugin_redirector: PluginRedirector
    validator_invoker: ValidatorInvoker

    simple_step_executor: SimpleStepExecutor
    agent_step_executor: AgentStepExecutor
    loop_step_executor: LoopStepExecutor
    parallel_step_executor: ParallelStepExecutor
    conditional_step_executor: ConditionalStepExecutor
    dynamic_router_step_executor: DynamicRouterStepExecutor
    hitl_step_executor: HitlStepExecutor
    cache_step_executor: CacheStepExecutor
    import_step_executor: ImportStepExecutor

    policy_registry: PolicyRegistry
    _dispatcher: ExecutionDispatcher
    _policy_default_step: "PolicyCallable"

    _agent_orchestrator: "AgentOrchestrator"
    _conditional_orchestrator: "ConditionalOrchestrator"
    _hitl_orchestrator: "HitlOrchestrator"
    _loop_orchestrator: "LoopOrchestrator"
    _pipeline_orchestrator: "PipelineOrchestrator"
    _import_orchestrator: "ImportOrchestrator"
    _validation_orchestrator: "ValidationOrchestrator"
    _context_setter: (
        Callable[[PipelineResult[DomainBaseModel], DomainBaseModel | None], None] | None
    )

    # Context variables moved to respective manager classes

    def __init__(
        self,
        agent_runner: IAgentRunner | None = None,
        processor_pipeline: IProcessorPipeline | None = None,
        validator_runner: IValidatorRunner | None = None,
        plugin_runner: IPluginRunner | None = None,
        usage_meter: IUsageMeter | None = None,
        quota_manager: QuotaManager | None = None,
        cache_backend: object | None = None,
        cache_key_generator: object | None = None,
        telemetry: ITelemetry | None = None,
        enable_cache: bool = True,
        # Additional parameters for compatibility
        serializer: ISerializer | None = None,
        hasher: IHasher | None = None,
        # UltraStepExecutor compatibility parameters
        cache_size: int = 1024,
        cache_ttl: int = 3600,
        concurrency_limit: int = 10,
        # Additional compatibility parameters
        # Injected policies
        timeout_runner: TimeoutRunner | None = None,
        unpacker: AgentResultUnpacker | None = None,
        plugin_redirector: PluginRedirector | None = None,
        validator_invoker: ValidatorInvoker | None = None,
        simple_step_executor: SimpleStepExecutor | None = None,
        agent_step_executor: AgentStepExecutor | None = None,
        loop_step_executor: LoopStepExecutor | None = None,
        parallel_step_executor: ParallelStepExecutor | None = None,
        conditional_step_executor: ConditionalStepExecutor | None = None,
        dynamic_router_step_executor: DynamicRouterStepExecutor | None = None,
        hitl_step_executor: HitlStepExecutor | None = None,
        cache_step_executor: CacheStepExecutor | None = None,
        usage_estimator: UsageEstimator | None = None,
        estimator_factory: UsageEstimatorFactory | None = None,
        policy_registry: PolicyRegistry | None = None,
        # Strict behavior toggles (robust defaults with optional enforcement)
        strict_context_isolation: bool = False,
        strict_context_merge: bool = False,
        enable_optimized_error_handling: bool = True,
        state_providers: dict[str, StateProvider[object]] | None = None,
        state_manager: StateManager[DomainBaseModel] | None = None,
        deps: ExecutorCoreDeps[TContext] | None = None,
        builder: FlujoRuntimeBuilder | None = None,
    ) -> None:
        # Validate parameters for compatibility
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if cache_ttl < 0:
            raise ValueError("cache_ttl must be non-negative")
        if concurrency_limit is not None and concurrency_limit <= 0:
            raise ValueError("concurrency_limit must be positive if specified")

        # Hardcode flag to False (standard handling)
        self.enable_optimized_error_handling = False

        builder_obj = builder or FlujoRuntimeBuilder()
        deps_obj = deps or builder_obj.build(
            agent_runner=agent_runner,
            processor_pipeline=processor_pipeline,
            validator_runner=validator_runner,
            plugin_runner=plugin_runner,
            usage_meter=usage_meter,
            telemetry=telemetry,
            quota_manager=quota_manager,
            cache_backend=cache_backend,
            cache_key_generator=cache_key_generator,
            serializer=serializer,
            hasher=hasher,
            enable_cache=enable_cache,
            cache_size=cache_size,
            cache_ttl=cache_ttl,
            fallback_handler=None,
            hydration_manager=None,
            background_task_manager=None,
            context_update_manager=None,
            step_history_tracker=None,
            estimator_factory=estimator_factory,
            usage_estimator=usage_estimator,
            timeout_runner=timeout_runner,
            unpacker=unpacker,
            plugin_redirector=plugin_redirector,
            validator_invoker=validator_invoker,
            simple_step_executor=simple_step_executor,
            agent_step_executor=agent_step_executor,
            loop_step_executor=loop_step_executor,
            parallel_step_executor=parallel_step_executor,
            conditional_step_executor=conditional_step_executor,
            dynamic_router_step_executor=dynamic_router_step_executor,
            hitl_step_executor=hitl_step_executor,
            cache_step_executor=cache_step_executor,
            agent_orchestrator=None,
            conditional_orchestrator=None,
            loop_orchestrator=None,
            hitl_orchestrator=None,
            import_orchestrator=None,
            pipeline_orchestrator=None,
            validation_orchestrator=None,
            state_providers=state_providers,
        )

        self._agent_runner = deps_obj.agent_runner
        self._processor_pipeline = deps_obj.processor_pipeline
        self._validator_runner = deps_obj.validator_runner
        self._plugin_runner = deps_obj.plugin_runner
        self._usage_meter = deps_obj.usage_meter
        self._cache_manager = deps_obj.cache_manager

        self._cache_backend = self._cache_manager.backend
        self._fallback_handler = deps_obj.fallback_handler
        self._telemetry = deps_obj.telemetry
        self._memory_store = deps_obj.memory_store
        self._memory_manager = getattr(deps_obj, "memory_manager", None)
        self._sandbox: SandboxProtocol = deps_obj.sandbox
        self._enable_cache = enable_cache
        # Estimation selection: factory first, then direct estimator, then default
        self._estimator_factory = deps_obj.estimator_factory
        self._usage_estimator = deps_obj.usage_estimator
        self._step_history_tracker = deps_obj.step_history_tracker
        self._quota_manager = deps_obj.quota_manager
        self._concurrency_limit = concurrency_limit

        self._serializer = deps_obj.serializer
        self._hasher = deps_obj.hasher
        self._cache_key_generator = deps_obj.cache_key_generator

        self._cache_locks: dict[str, asyncio.Lock] = {}
        self._cache_locks_lock: asyncio.Lock | None = None

        # Strict behavior settings (defaults can be overridden by global settings)
        self._strict_context_isolation = (
            bool(strict_context_isolation) or _DEFAULT_STRICT_CONTEXT_ISOLATION
        )
        self._strict_context_merge = bool(strict_context_merge) or _DEFAULT_STRICT_CONTEXT_MERGE
        self._hydration_manager = deps_obj.hydration_manager
        self._hydration_manager.set_telemetry(self._telemetry)
        self._background_task_manager = deps_obj.background_task_manager
        self._bg_task_handler = deps_obj.bg_task_handler
        self.state_manager = state_manager
        self._context_update_manager = deps_obj.context_update_manager
        self._agent_orchestrator = deps_obj.agent_orchestrator
        self._conditional_orchestrator = deps_obj.conditional_orchestrator
        self._hitl_orchestrator = deps_obj.hitl_orchestrator
        self._loop_orchestrator = deps_obj.loop_orchestrator
        self._pipeline_orchestrator = deps_obj.pipeline_orchestrator
        self._validation_orchestrator = deps_obj.validation_orchestrator

        self._state_providers = self._hydration_manager._state_providers

        # Assign policies
        self.timeout_runner = deps_obj.timeout_runner
        self.unpacker = deps_obj.unpacker
        self.plugin_redirector = deps_obj.plugin_redirector
        self.validator_invoker = deps_obj.validator_invoker
        self.simple_step_executor = deps_obj.simple_step_executor
        self.agent_step_executor = deps_obj.agent_step_executor
        self.loop_step_executor = deps_obj.loop_step_executor
        self.parallel_step_executor = deps_obj.parallel_step_executor
        self.conditional_step_executor = deps_obj.conditional_step_executor
        self.dynamic_router_step_executor = deps_obj.dynamic_router_step_executor
        self.hitl_step_executor = deps_obj.hitl_step_executor
        self.cache_step_executor = deps_obj.cache_step_executor
        self.import_step_executor = deps_obj.import_step_executor
        self._import_orchestrator = deps_obj.import_orchestrator

        # FSD-010: Initialize and populate the policy registry used for dispatch
        registry_factory = deps_obj.policy_registry_factory
        self.policy_registry = policy_registry or (
            registry_factory(self)
            if registry_factory is not None
            else create_default_registry(self)
        )

        # Policy handlers (delegated for composition-root slimming)
        policy_handlers_factory = deps_obj.policy_handlers_factory
        self._policy_handlers = (
            policy_handlers_factory(self)
            if policy_handlers_factory is not None
            else PolicyHandlers(self)
        )
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
        dispatcher_factory = deps_obj.dispatcher_factory
        self._dispatcher = (
            dispatcher_factory(self.policy_registry, self)
            if dispatcher_factory is not None
            else ExecutionDispatcher(self.policy_registry, core=self)
        )
        dispatch_handler_factory = deps_obj.dispatch_handler_factory
        self._dispatch_handler = (
            dispatch_handler_factory(self)
            if dispatch_handler_factory is not None
            else DispatchHandler(self)
        )
        result_handler_factory = deps_obj.result_handler_factory
        self._result_handler = (
            result_handler_factory(self)
            if result_handler_factory is not None
            else ResultHandler(self)
        )
        telemetry_handler_factory = deps_obj.telemetry_handler_factory
        self._telemetry_handler = (
            telemetry_handler_factory(self)
            if telemetry_handler_factory is not None
            else TelemetryHandler(self)
        )
        step_handler_factory = deps_obj.step_handler_factory
        self._step_handler = (
            step_handler_factory(self) if step_handler_factory is not None else StepHandler(self)
        )
        agent_handler_factory = deps_obj.agent_handler_factory
        self._agent_handler = (
            agent_handler_factory(self) if agent_handler_factory is not None else AgentHandler(self)
        )
        self._governance_engine = deps_obj.governance_engine
        self._shadow_evaluator: ShadowEvaluator | None = getattr(deps_obj, "shadow_evaluator", None)

        # Initialize orchestrators that depend on executors registered above

    @property
    def state_manager(self) -> "StateManager[DomainBaseModel] | None":
        return getattr(self, "_state_manager", None)

    @state_manager.setter
    def state_manager(self, value: "StateManager[DomainBaseModel] | None") -> None:
        object.__setattr__(self, "_state_manager", value)
        try:
            bg_handler = getattr(self, "_bg_task_handler", None)
            if bg_handler is not None:
                bg_handler.state_manager = value
        except Exception:
            pass

    def _get_cache_locks_lock(self) -> asyncio.Lock:
        """Lazily create the cache locks lock on first access."""
        if self._cache_locks_lock is None:
            self._cache_locks_lock = asyncio.Lock()
        return self._cache_locks_lock

    @property
    def cache(self) -> _LRUCache:
        return self._cache_manager.get_internal_cache()

    @property
    def memory_store(self) -> VectorStoreProtocol:
        """Exposed vector store used for long-term memory; defaults to a Null store."""
        return self._memory_store

    @property
    def memory_manager(self) -> object | None:
        """Exposed memory manager; may be NullMemoryManager when disabled."""
        return self._memory_manager

    @property
    def sandbox(self) -> SandboxProtocol:
        """Exposed sandbox implementation used for code execution; defaults to Null sandbox."""
        return self._sandbox

    async def clear_cache(self) -> None:
        """Async version - use this in async contexts with await."""
        await self._cache_manager.clear_cache()

    def clear_cache_sync(self) -> None:
        """Synchronous version - maintains backward compatibility for synchronous callers.

        This method can be called from synchronous code. It will:
        - Use asyncio.run() if no event loop is running
        - Raise RuntimeError if called from within a running event loop (use await clear_cache() instead)
        """
        msg = (
            "clear_cache_sync() cannot be called from within a running event loop. "
            "Use 'await executor.clear_cache()' instead."
        )
        try:
            run_sync(self._cache_manager.clear_cache(), running_loop_error=msg)
        except TypeError:
            raise RuntimeError(msg) from None

    def _cache_key(self, frame: ExecutionFrame[TFrameContext]) -> str:
        return self._cache_manager.generate_cache_key(
            frame.step, frame.data, frame.context, getattr(frame, "resources", None)
        )

    def _cache_enabled(self) -> bool:
        """Return whether cache is enabled, honoring task-local overrides."""
        override = _CACHE_OVERRIDE.get(None)
        if override is not None:
            return bool(override)
        return bool(self._enable_cache)

    _normalize_frame_context = staticmethod(normalize_frame_context)

    async def _set_quota_and_hydrate(self, frame: ExecutionFrame[TContext]) -> None:
        """Assign quota to the execution context and hydrate managed state."""
        await set_quota_and_hydrate(frame, self._quota_manager, self._hydration_manager)

    def _handle_missing_agent_exception(
        self, err: MissingAgentError, step: object, *, called_with_frame: bool
    ) -> StepOutcome[StepResult] | StepResult:
        return handle_missing_agent_exception(self, err, step, called_with_frame=called_with_frame)

    async def _persist_and_finalize(
        self,
        *,
        step: object,
        result: StepResult,
        cache_key: str | None,
        called_with_frame: bool,
        frame: ExecutionFrame[TContext] | None = None,
    ) -> StepOutcome[StepResult] | StepResult:
        return await persist_and_finalize(
            self,
            step=step,
            result=result,
            cache_key=cache_key,
            called_with_frame=called_with_frame,
            frame=frame,
        )

    def _handle_unexpected_exception(
        self,
        *,
        step: object,
        frame: ExecutionFrame[TContext],
        exc: Exception,
        called_with_frame: bool,
    ) -> StepOutcome[StepResult] | StepResult:
        return handle_unexpected_exception(
            self, step=step, frame=frame, exc=exc, called_with_frame=called_with_frame
        )

    async def _maybe_use_cache(
        self, frame: ExecutionFrame[TContext], *, called_with_frame: bool
    ) -> tuple[StepOutcome[StepResult] | StepResult | None, str | None]:
        return await maybe_use_cache(self, frame, called_with_frame=called_with_frame)

    def _get_current_quota(self) -> Quota | None:
        """Best-effort getter for the current quota using the manager first."""
        return get_current_quota(self._quota_manager)

    def _set_current_quota(self, quota: Quota | None) -> object | None:
        """Best-effort setter for the current quota (returns token when available)."""
        return set_current_quota(self._quota_manager, quota)

    def _reset_current_quota(self, token: object | None) -> None:
        """Best-effort reset for quota context tokens."""
        reset_current_quota(self._quota_manager, token)

    def _get_background_quota(self, parent_quota: Quota | None = None) -> Quota | None:
        return self._bg_task_handler.get_background_quota(parent_quota)

    async def _register_background_task(
        self,
        *,
        task_id: str,
        bg_run_id: str,
        parent_run_id: str | None,
        step_name: str,
        data: object,
        context: TContext | None,
        metadata: JSONObject | None = None,
    ) -> None:
        await self._bg_task_handler.register_background_task(
            task_id=task_id,
            bg_run_id=bg_run_id,
            parent_run_id=parent_run_id,
            step_name=step_name,
            data=data,
            context=context,
            metadata=metadata,
        )

    async def _mark_background_task_completed(
        self,
        *,
        task_id: str,
        context: TContext | None,
        metadata: JSONObject | None = None,
    ) -> None:
        await self._bg_task_handler.mark_background_task_completed(
            task_id=task_id, context=context, metadata=metadata
        )

    async def _mark_background_task_failed(
        self,
        *,
        task_id: str,
        context: TContext | None,
        error: Exception,
        metadata: JSONObject | None = None,
    ) -> None:
        await self._bg_task_handler.mark_background_task_failed(
            task_id=task_id, context=context, error=error, metadata=metadata
        )

    async def _mark_background_task_paused(
        self,
        *,
        task_id: str,
        context: TContext | None,
        error: Exception,
        metadata: JSONObject | None = None,
    ) -> None:
        await self._bg_task_handler.mark_background_task_paused(
            task_id=task_id, context=context, error=error, metadata=metadata
        )

    def _hash_obj(self, obj: object) -> str:
        return hash_obj(obj, self._serializer, self._hasher)

    def _isolate_context(self, context: TContext | None) -> TContext | None:
        return isolate_context(
            context, strict_context_isolation=bool(self._strict_context_isolation)
        )

    def _merge_context_updates(
        self,
        main_context: TContext | None,
        branch_context: TContext | None,
    ) -> TContext | None:
        return merge_context_updates(
            main_context,
            branch_context,
            strict_context_merge=bool(self._strict_context_merge),
        )

    def _accumulate_loop_context(
        self,
        current_context: TContext | None,
        iteration_context: TContext | None,
    ) -> TContext | None:
        return accumulate_loop_context(
            current_context,
            iteration_context,
            strict_context_merge=bool(self._strict_context_merge),
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
        self, frame: ExecutionFrame[TContext], *, called_with_frame: bool
    ) -> StepOutcome[StepResult] | StepResult:
        return await self._dispatch_handler.dispatch(frame, called_with_frame=called_with_frame)

    async def _execute_complex_step(
        self,
        *,
        step: object,
        data: object,
        context: TContext | None,
        resources: object | None,
        limits: UsageLimits | None,
        stream: bool = False,
        on_chunk: Callable[[object], Awaitable[None]] | None = None,
        context_setter: Callable[[PipelineResult[DomainBaseModel], DomainBaseModel | None], None]
        | None = None,
        _fallback_depth: int = 0,
    ) -> StepResult:
        try:
            fb_depth = int(_fallback_depth)
        except Exception:
            fb_depth = 0
        outcome = await self.execute(
            step,
            data,
            context=context,
            resources=resources,
            limits=limits,
            stream=stream,
            on_chunk=on_chunk,
            context_setter=context_setter,
            _fallback_depth=fb_depth,
        )
        return self._unwrap_outcome_to_step_result(outcome, self._safe_step_name(step))

    async def wait_for_background_tasks(self, timeout: float = 5.0) -> None:
        """Wait for all background tasks to complete with a timeout."""
        await self._background_task_manager.wait_for_completion(timeout)

    async def aclose(self) -> None:
        """Best-effort cleanup of executor-owned resources."""
        try:
            await self.wait_for_background_tasks()
        except Exception:
            pass

        # Close memory indexing manager first (flush pending tasks)
        try:
            if self._memory_manager is not None and hasattr(self._memory_manager, "close"):
                res = self._memory_manager.close()
                if asyncio.iscoroutine(res):
                    await res
        except Exception:
            pass

        # Close vector store if it exposes close/cleanup
        try:
            if self._memory_store is not None and hasattr(self._memory_store, "close"):
                res = self._memory_store.close()
                if asyncio.iscoroutine(res):
                    await res
        except Exception:
            pass

    async def execute(
        self,
        frame_or_step: ExecutionFrame[TContext] | object | None = None,
        data: object | None = None,
        *args: object,
        **kwargs: object,
    ) -> StepOutcome[StepResult] | StepResult:
        """Public entrypoint that delegates to the shared execution flow."""
        return await execute_entrypoint(self, frame_or_step, data, *args, **kwargs)

    def _log_execution_error(self, step_name: str, exc: Exception) -> None:
        log_execution_error(self, step_name, exc)

    _build_failure_outcome = build_failure

    _make_execution_frame = make_execution_frame

    # Preserve failure builder compatibility
    _failure_builder = build_failure_outcome

    # Compatibility shims retained for legacy call sites/tests
    async def _execute_pipeline_via_policies(
        self,
        pipeline: object,
        data: object,
        context: TContext | None,
        resources: object | None,
        limits: UsageLimits | None,
        context_setter: Callable[[PipelineResult[DomainBaseModel], DomainBaseModel | None], None]
        | None = None,
    ) -> PipelineResult[DomainBaseModel]:
        return await self._step_handler.pipeline(
            pipeline, data, context, resources, limits, context_setter
        )

    async def _execute_pipeline(
        self,
        pipeline: object,
        data: object,
        context: TContext,
        resources: object,
        limits: UsageLimits,
        context_setter: Callable[[PipelineResult[DomainBaseModel], DomainBaseModel | None], None]
        | None,
    ) -> PipelineResult[DomainBaseModel]:
        return await self._step_handler.pipeline(
            pipeline, data, context, resources, limits, context_setter
        )

    async def _handle_loop_step(
        self,
        loop_step: object,
        data: object,
        context: TContext,
        resources: object,
        limits: UsageLimits,
        context_setter: Callable[[PipelineResult[DomainBaseModel], DomainBaseModel | None], None]
        | None,
        _fallback_depth: int = 0,
    ) -> StepResult:
        return await self._step_handler.loop_step(
            loop_step, data, context, resources, limits, context_setter, _fallback_depth
        )

    async def _execute_loop(
        self,
        loop_step: object,
        data: object,
        context: TContext,
        resources: object,
        limits: UsageLimits,
        context_setter: Callable[[PipelineResult[DomainBaseModel], DomainBaseModel | None], None]
        | None,
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
        step: object,
        data: object,
        context: TContext | None,
        resources: object | None,
        limits: UsageLimits | None,
        context_setter: Callable[[PipelineResult[DomainBaseModel], DomainBaseModel | None], None]
        | None = None,
        **_: object,
    ) -> StepResult:
        return await self._step_handler.cache_step(
            step, data, context, resources, limits, context_setter
        )

    async def _handle_conditional_step(
        self,
        step: object,
        data: object,
        context: TContext | None,
        resources: object | None,
        limits: UsageLimits | None,
        context_setter: Callable[[PipelineResult[DomainBaseModel], DomainBaseModel | None], None]
        | None,
        _fallback_depth: int = 0,
        **_: object,
    ) -> StepResult:
        return await self._step_handler.conditional_step(
            step, data, context, resources, limits, context_setter, _fallback_depth
        )

    async def _handle_dynamic_router_step(
        self,
        step: object,
        data: object,
        context: TContext | None,
        resources: object | None,
        limits: UsageLimits | None,
        context_setter: Callable[[PipelineResult[DomainBaseModel], DomainBaseModel | None], None]
        | None = None,
        **_: object,
    ) -> StepResult:
        return await self._step_handler.dynamic_router_step(
            step, data, context, resources, limits, context_setter
        )

    async def _handle_hitl_step(
        self,
        step: object,
        data: object,
        context: TContext | None,
        resources: object | None,
        limits: UsageLimits | None,
        context_setter: Callable[[PipelineResult[DomainBaseModel], DomainBaseModel | None], None]
        | None,
        stream: bool = False,
        on_chunk: Callable[[object], Awaitable[None]] | None = None,
        cache_key: str | None = None,
        _fallback_depth: int = 0,
        **_: object,
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

    async def _handle_parallel_step(
        self,
        step: object | None = None,
        data: object | None = None,
        context: TContext | None = None,
        resources: object | None = None,
        limits: UsageLimits | None = None,
        context_setter: Callable[[PipelineResult[DomainBaseModel], DomainBaseModel | None], None]
        | None = None,
        *,
        parallel_step: object | None = None,
    ) -> StepResult:
        ps = parallel_step if parallel_step is not None else step
        return await self._step_handler.parallel_step(
            ps, data, context, resources, limits, context_setter
        )

    async def _handle_dynamic_router(
        self,
        step: object | None = None,
        data: object | None = None,
        context: TContext | None = None,
        resources: object | None = None,
        limits: UsageLimits | None = None,
        context_setter: Callable[[PipelineResult[DomainBaseModel], DomainBaseModel | None], None]
        | None = None,
        router_step: object | None = None,
    ) -> StepResult:
        rs = router_step if router_step is not None else step
        return await self._step_handler.dynamic_router_wrapper(
            step, data, context, resources, limits, context_setter, rs
        )

    async def _execute_agent_with_orchestration(
        self,
        step: object,
        data: object,
        context: TContext | None,
        resources: object | None,
        limits: UsageLimits | None,
        stream: bool,
        on_chunk: Callable[[object], Awaitable[None]] | None,
        cache_key: str | None,
        _fallback_depth: int,
    ) -> StepOutcome[StepResult]:
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

    _safe_step_name = staticmethod(safe_step_name)
    _format_feedback = staticmethod(format_feedback)


__all__ = [
    "ExecutorCore",
    "PluginError",
    "StepExecutor",
    "_UsageTracker",
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
