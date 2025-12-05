"""Runtime builder for ExecutorCore dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

from .agent_handler import AgentHandler
from .agent_orchestrator import AgentOrchestrator
from .background_task_manager import BackgroundTaskManager
from .cache_manager import CacheManager
from .conditional_orchestrator import ConditionalOrchestrator
from .context_update_manager import ContextUpdateManager
from .dispatch_handler import DispatchHandler
from .default_cache_components import (
    Blake3Hasher,
    DefaultCacheKeyGenerator,
    InMemoryLRUBackend,
    OrjsonSerializer,
    ThreadSafeMeter,
)
from .default_components import (
    DefaultAgentRunner,
    DefaultPluginRunner,
    DefaultProcessorPipeline,
    DefaultTelemetry,
    DefaultValidatorRunner,
)
from .estimation import (
    HeuristicUsageEstimator,
    UsageEstimator,
    UsageEstimatorFactory,
    build_default_estimator_factory,
)
from .fallback_handler import FallbackHandler
from .governance_policy import (
    AllowAllGovernancePolicy,
    DenyAllGovernancePolicy,
    GovernanceEngine,
    GovernancePolicy,
)
from .execution_dispatcher import ExecutionDispatcher
from .hitl_orchestrator import HitlOrchestrator
from .hydration_manager import HydrationManager
from .import_orchestrator import ImportOrchestrator
from .loop_orchestrator import LoopOrchestrator
from .policy_handlers import PolicyHandlers
from .policy_registry import PolicyRegistry, create_default_registry
from .pipeline_orchestrator import PipelineOrchestrator
from .quota_manager import QuotaManager
from .result_handler import ResultHandler
from .step_history_tracker import StepHistoryTracker
from .step_handler import StepHandler
from .telemetry_handler import TelemetryHandler
from .validation_orchestrator import ValidationOrchestrator
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
from ...utils.config import get_settings

if TYPE_CHECKING:  # pragma: no cover
    from .executor_core import ExecutorCore


@dataclass
class ExecutorCoreDeps:
    """Container for ExecutorCore injectables."""

    agent_runner: Any
    processor_pipeline: Any
    validator_runner: Any
    plugin_runner: Any
    usage_meter: Any
    telemetry: Any
    quota_manager: QuotaManager
    cache_manager: CacheManager
    serializer: Any
    hasher: Any
    cache_key_generator: Any
    fallback_handler: FallbackHandler
    hydration_manager: HydrationManager
    background_task_manager: BackgroundTaskManager
    context_update_manager: ContextUpdateManager
    step_history_tracker: StepHistoryTracker
    estimator_factory: UsageEstimatorFactory
    usage_estimator: UsageEstimator
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
    import_step_executor: Any
    agent_orchestrator: AgentOrchestrator
    conditional_orchestrator: ConditionalOrchestrator
    loop_orchestrator: LoopOrchestrator
    hitl_orchestrator: HitlOrchestrator
    import_orchestrator: ImportOrchestrator
    pipeline_orchestrator: PipelineOrchestrator
    validation_orchestrator: ValidationOrchestrator
    policy_registry_factory: Callable[["ExecutorCore[Any]"], PolicyRegistry] | None = None
    policy_handlers_factory: Callable[["ExecutorCore[Any]"], PolicyHandlers] | None = None
    dispatcher_factory: (
        Callable[[PolicyRegistry, "ExecutorCore[Any]"], ExecutionDispatcher] | None
    ) = None
    dispatch_handler_factory: Callable[["ExecutorCore[Any]"], DispatchHandler] | None = None
    result_handler_factory: Callable[["ExecutorCore[Any]"], ResultHandler] | None = None
    telemetry_handler_factory: Callable[["ExecutorCore[Any]"], TelemetryHandler] | None = None
    step_handler_factory: Callable[["ExecutorCore[Any]"], StepHandler] | None = None
    agent_handler_factory: Callable[["ExecutorCore[Any]"], AgentHandler] | None = None
    governance_engine: GovernanceEngine | None = None


class FlujoRuntimeBuilder:
    """Factory that wires ExecutorCore dependencies with overridable defaults."""

    def build(
        self,
        *,
        agent_runner: Optional[Any] = None,
        processor_pipeline: Optional[Any] = None,
        validator_runner: Optional[Any] = None,
        plugin_runner: Optional[Any] = None,
        usage_meter: Optional[Any] = None,
        telemetry: Optional[Any] = None,
        quota_manager: Optional[QuotaManager] = None,
        cache_backend: Any = None,
        cache_key_generator: Any = None,
        serializer: Any = None,
        hasher: Any = None,
        enable_cache: bool = True,
        cache_size: int = 1024,
        cache_ttl: int = 3600,
        fallback_handler: Optional[FallbackHandler] = None,
        hydration_manager: Optional[HydrationManager] = None,
        background_task_manager: Optional[BackgroundTaskManager] = None,
        context_update_manager: Optional[ContextUpdateManager] = None,
        step_history_tracker: Optional[StepHistoryTracker] = None,
        estimator_factory: Optional[UsageEstimatorFactory] = None,
        usage_estimator: Optional[UsageEstimator] = None,
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
        import_step_executor: Optional[Any] = None,
        agent_orchestrator: Optional[AgentOrchestrator] = None,
        conditional_orchestrator: Optional[ConditionalOrchestrator] = None,
        loop_orchestrator: Optional[LoopOrchestrator] = None,
        hitl_orchestrator: Optional[HitlOrchestrator] = None,
        import_orchestrator: Optional[ImportOrchestrator] = None,
        pipeline_orchestrator: Optional[PipelineOrchestrator] = None,
        validation_orchestrator: Optional[ValidationOrchestrator] = None,
        state_providers: Optional[dict[str, Any]] = None,
        policy_registry_factory: Callable[["ExecutorCore[Any]"], PolicyRegistry] | None = None,
        policy_handlers_factory: Callable[["ExecutorCore[Any]"], PolicyHandlers] | None = None,
        dispatcher_factory: Callable[[PolicyRegistry, "ExecutorCore[Any]"], ExecutionDispatcher]
        | None = None,
        dispatch_handler_factory: Callable[["ExecutorCore[Any]"], DispatchHandler] | None = None,
        result_handler_factory: Callable[["ExecutorCore[Any]"], ResultHandler] | None = None,
        telemetry_handler_factory: Callable[["ExecutorCore[Any]"], TelemetryHandler] | None = None,
        step_handler_factory: Callable[["ExecutorCore[Any]"], StepHandler] | None = None,
        agent_handler_factory: Callable[["ExecutorCore[Any]"], AgentHandler] | None = None,
        governance_policies: tuple[GovernancePolicy, ...] | None = None,
    ) -> ExecutorCoreDeps:
        serializer_obj = serializer or OrjsonSerializer()
        hasher_obj = hasher or Blake3Hasher()
        cache_key_gen = cache_key_generator or DefaultCacheKeyGenerator(hasher_obj)
        backend = cache_backend or InMemoryLRUBackend(max_size=cache_size, ttl_s=cache_ttl)
        cache_manager = CacheManager(
            backend=backend,
            key_generator=cache_key_gen,
            enable_cache=enable_cache,
        )

        plugin_runner_obj = plugin_runner or DefaultPluginRunner()
        agent_runner_obj = agent_runner or DefaultAgentRunner()
        validator_runner_obj = validator_runner or DefaultValidatorRunner()

        timeout_runner_obj: TimeoutRunner = timeout_runner or DefaultTimeoutRunner()
        unpacker_obj: AgentResultUnpacker = unpacker or DefaultAgentResultUnpacker()
        plugin_redirector_obj: PluginRedirector = plugin_redirector or DefaultPluginRedirector(
            plugin_runner_obj, agent_runner_obj
        )
        validator_invoker_obj: ValidatorInvoker = validator_invoker or DefaultValidatorInvoker(
            validator_runner_obj
        )
        simple_step_executor_obj: SimpleStepExecutor = cast(
            SimpleStepExecutor, simple_step_executor or DefaultSimpleStepExecutor()
        )
        agent_step_executor_obj: AgentStepExecutor = (
            agent_step_executor or DefaultAgentStepExecutor()
        )
        loop_step_executor_obj: LoopStepExecutor = loop_step_executor or DefaultLoopStepExecutor()
        parallel_step_executor_obj: ParallelStepExecutor = (
            parallel_step_executor or DefaultParallelStepExecutor()
        )
        conditional_step_executor_obj: ConditionalStepExecutor = (
            conditional_step_executor or DefaultConditionalStepExecutor()
        )
        dynamic_router_step_executor_obj: DynamicRouterStepExecutor = (
            dynamic_router_step_executor or DefaultDynamicRouterStepExecutor()
        )
        hitl_step_executor_obj: HitlStepExecutor = hitl_step_executor or DefaultHitlStepExecutor()
        cache_step_executor_obj: CacheStepExecutor = (
            cache_step_executor or DefaultCacheStepExecutor()
        )

        import_step_executor_obj = import_step_executor
        if import_step_executor_obj is None:
            from .step_policies import DefaultImportStepExecutor

            import_step_executor_obj = DefaultImportStepExecutor()

        policy_registry_factory_obj = policy_registry_factory or (
            lambda core: create_default_registry(core)
        )
        policy_handlers_factory_obj = policy_handlers_factory or (lambda core: PolicyHandlers(core))
        dispatcher_factory_obj = dispatcher_factory or (
            lambda registry, core: ExecutionDispatcher(registry, core=core)
        )
        dispatch_handler_factory_obj = dispatch_handler_factory or (
            lambda core: DispatchHandler(core)
        )
        result_handler_factory_obj = result_handler_factory or (lambda core: ResultHandler(core))
        telemetry_handler_factory_obj = telemetry_handler_factory or (
            lambda core: TelemetryHandler(core)
        )
        step_handler_factory_obj = step_handler_factory or (lambda core: StepHandler(core))
        agent_handler_factory_obj = agent_handler_factory or (lambda core: AgentHandler(core))
        settings = get_settings()
        policies = governance_policies
        if policies is None:
            mode = getattr(settings, "governance_mode", "allow_all")
            if mode == "deny_all":
                policies = (DenyAllGovernancePolicy(),)
            else:
                policies = (AllowAllGovernancePolicy(),)
        governance_engine_obj = GovernanceEngine(policies=policies)

        return ExecutorCoreDeps(
            agent_runner=agent_runner_obj,
            processor_pipeline=processor_pipeline or DefaultProcessorPipeline(),
            validator_runner=validator_runner_obj,
            plugin_runner=plugin_runner_obj,
            usage_meter=usage_meter or ThreadSafeMeter(),
            telemetry=telemetry or DefaultTelemetry(),
            quota_manager=quota_manager or QuotaManager(),
            cache_manager=cache_manager,
            serializer=serializer_obj,
            hasher=hasher_obj,
            cache_key_generator=cache_key_gen,
            fallback_handler=fallback_handler or FallbackHandler(),
            hydration_manager=hydration_manager or HydrationManager(state_providers),
            background_task_manager=background_task_manager or BackgroundTaskManager(),
            context_update_manager=context_update_manager or ContextUpdateManager(),
            step_history_tracker=step_history_tracker or StepHistoryTracker(),
            estimator_factory=estimator_factory or build_default_estimator_factory(),
            usage_estimator=usage_estimator or HeuristicUsageEstimator(),
            timeout_runner=timeout_runner_obj,
            unpacker=unpacker_obj,
            plugin_redirector=plugin_redirector_obj,
            validator_invoker=validator_invoker_obj,
            simple_step_executor=simple_step_executor_obj,
            agent_step_executor=agent_step_executor_obj,
            loop_step_executor=loop_step_executor_obj,
            parallel_step_executor=parallel_step_executor_obj,
            conditional_step_executor=conditional_step_executor_obj,
            dynamic_router_step_executor=dynamic_router_step_executor_obj,
            hitl_step_executor=hitl_step_executor_obj,
            cache_step_executor=cache_step_executor_obj,
            import_step_executor=import_step_executor_obj,
            agent_orchestrator=agent_orchestrator
            or AgentOrchestrator(plugin_runner=plugin_runner_obj),
            conditional_orchestrator=conditional_orchestrator or ConditionalOrchestrator(),
            loop_orchestrator=loop_orchestrator or LoopOrchestrator(),
            hitl_orchestrator=hitl_orchestrator or HitlOrchestrator(),
            import_orchestrator=import_orchestrator or ImportOrchestrator(import_step_executor_obj),
            pipeline_orchestrator=pipeline_orchestrator or PipelineOrchestrator(),
            validation_orchestrator=validation_orchestrator or ValidationOrchestrator(),
            policy_registry_factory=policy_registry_factory_obj,
            policy_handlers_factory=policy_handlers_factory_obj,
            dispatcher_factory=dispatcher_factory_obj,
            dispatch_handler_factory=dispatch_handler_factory_obj,
            result_handler_factory=result_handler_factory_obj,
            telemetry_handler_factory=telemetry_handler_factory_obj,
            step_handler_factory=step_handler_factory_obj,
            agent_handler_factory=agent_handler_factory_obj,
            governance_engine=governance_engine_obj,
        )
