"""Runtime builder for ExecutorCore dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generic, Optional, TypeGuard
from urllib.parse import urlparse
import importlib

from ..agent_handler import AgentHandler
from ..agent_orchestrator import AgentOrchestrator
from .background_task_manager import BackgroundTaskManager
from .background_task_handler import BackgroundTaskHandler
from .cache_manager import CacheManager
from ..conditional_orchestrator import ConditionalOrchestrator
from ..context_update_manager import ContextUpdateManager
from ..dispatch_handler import DispatchHandler
from ..executor_protocols import (
    IAgentRunner,
    IHasher,
    IPluginRunner,
    IProcessorPipeline,
    ISerializer,
    ITelemetry,
    IUsageMeter,
    IValidatorRunner,
)
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
from ..governance_policy import (
    AllowAllGovernancePolicy,
    DenyAllGovernancePolicy,
    GovernanceEngine,
    GovernancePolicy,
    PIIScrubbingPolicy,
    ToolAllowlistPolicy,
)
from ..execution_dispatcher import ExecutionDispatcher
from ..hitl_orchestrator import HitlOrchestrator
from .hydration_manager import HydrationManager
from ..import_orchestrator import ImportOrchestrator
from ..loop_orchestrator import LoopOrchestrator
from ..policy_handlers import PolicyHandlers
from ..policy_registry import PolicyRegistry, create_default_registry
from ..pipeline_orchestrator import PipelineOrchestrator
from .quota_manager import QuotaManager
from ..result_handler import ResultHandler
from .shadow_evaluator import ShadowEvalConfig, ShadowEvaluator
from ..step_history_tracker import StepHistoryTracker
from ..step_handler import StepHandler
from .telemetry_handler import TelemetryHandler
from ..validation_orchestrator import ValidationOrchestrator
from ..types import TContext
from ..step_policies import (
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
    ImportStepExecutor,
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
from ....domain.memory import VectorStoreProtocol
from ....domain.interfaces import StateProvider
from ....domain.sandbox import SandboxProtocol
from ....infra.memory import (
    NullVectorStore,
    MemoryManager,
    NullMemoryManager,
    SQLiteVectorStore,
    PostgresVectorStore,
)
from ....infra.sandbox import NullSandbox, RemoteSandbox, DockerSandbox
from ....utils.config import get_settings
from ....infra.config_manager import get_state_uri

if TYPE_CHECKING:  # pragma: no cover
    from ..executor_core import ExecutorCore


@dataclass
class ExecutorCoreDeps(Generic[TContext]):
    """Container for ExecutorCore injectables."""

    agent_runner: IAgentRunner
    processor_pipeline: IProcessorPipeline
    validator_runner: IValidatorRunner
    plugin_runner: IPluginRunner
    usage_meter: IUsageMeter
    telemetry: ITelemetry
    quota_manager: QuotaManager
    cache_manager: CacheManager
    serializer: ISerializer
    hasher: IHasher
    cache_key_generator: object
    fallback_handler: FallbackHandler
    hydration_manager: HydrationManager
    memory_store: VectorStoreProtocol
    sandbox: SandboxProtocol
    memory_manager: MemoryManager
    background_task_manager: BackgroundTaskManager
    bg_task_handler: BackgroundTaskHandler
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
    import_step_executor: ImportStepExecutor
    agent_orchestrator: AgentOrchestrator
    conditional_orchestrator: ConditionalOrchestrator
    loop_orchestrator: LoopOrchestrator
    hitl_orchestrator: HitlOrchestrator
    import_orchestrator: ImportOrchestrator
    pipeline_orchestrator: PipelineOrchestrator
    validation_orchestrator: ValidationOrchestrator
    policy_registry_factory: Callable[["ExecutorCore[TContext]"], PolicyRegistry] | None = None
    policy_handlers_factory: (
        Callable[["ExecutorCore[TContext]"], PolicyHandlers[TContext]] | None
    ) = None
    dispatcher_factory: (
        Callable[[PolicyRegistry, "ExecutorCore[TContext]"], ExecutionDispatcher] | None
    ) = None
    dispatch_handler_factory: Callable[["ExecutorCore[TContext]"], DispatchHandler] | None = None
    result_handler_factory: Callable[["ExecutorCore[TContext]"], ResultHandler] | None = None
    telemetry_handler_factory: Callable[["ExecutorCore[TContext]"], TelemetryHandler] | None = None
    step_handler_factory: Callable[["ExecutorCore[TContext]"], StepHandler] | None = None
    agent_handler_factory: Callable[["ExecutorCore[TContext]"], AgentHandler] | None = None
    governance_engine: GovernanceEngine | None = None
    shadow_evaluator: ShadowEvaluator | None = None


class FlujoRuntimeBuilder:
    """Factory that wires ExecutorCore dependencies with overridable defaults."""

    def build(
        self,
        *,
        agent_runner: Optional[IAgentRunner] = None,
        processor_pipeline: Optional[IProcessorPipeline] = None,
        validator_runner: Optional[IValidatorRunner] = None,
        plugin_runner: Optional[IPluginRunner] = None,
        usage_meter: Optional[IUsageMeter] = None,
        telemetry: Optional[ITelemetry] = None,
        quota_manager: Optional[QuotaManager] = None,
        cache_backend: object | None = None,
        cache_key_generator: object | None = None,
        serializer: Optional[ISerializer] = None,
        hasher: Optional[IHasher] = None,
        enable_cache: bool = True,
        cache_size: int = 1024,
        cache_ttl: int = 3600,
        fallback_handler: Optional[FallbackHandler] = None,
        hydration_manager: Optional[HydrationManager] = None,
        sandbox: Optional[SandboxProtocol] = None,
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
        import_step_executor: Optional[ImportStepExecutor] = None,
        agent_orchestrator: Optional[AgentOrchestrator] = None,
        conditional_orchestrator: Optional[ConditionalOrchestrator] = None,
        loop_orchestrator: Optional[LoopOrchestrator] = None,
        hitl_orchestrator: Optional[HitlOrchestrator] = None,
        import_orchestrator: Optional[ImportOrchestrator] = None,
        pipeline_orchestrator: Optional[PipelineOrchestrator] = None,
        validation_orchestrator: Optional[ValidationOrchestrator] = None,
        state_providers: Optional[dict[str, StateProvider[object]]] = None,
        memory_store: Optional[VectorStoreProtocol] = None,
        policy_registry_factory: (
            Callable[["ExecutorCore[TContext]"], PolicyRegistry] | None
        ) = None,
        policy_handlers_factory: (
            Callable[["ExecutorCore[TContext]"], PolicyHandlers[TContext]] | None
        ) = None,
        dispatcher_factory: Callable[
            [PolicyRegistry, "ExecutorCore[TContext]"], ExecutionDispatcher
        ]
        | None = None,
        dispatch_handler_factory: (
            Callable[["ExecutorCore[TContext]"], DispatchHandler] | None
        ) = None,
        result_handler_factory: (Callable[["ExecutorCore[TContext]"], ResultHandler] | None) = None,
        telemetry_handler_factory: (
            Callable[["ExecutorCore[TContext]"], TelemetryHandler] | None
        ) = None,
        step_handler_factory: (Callable[["ExecutorCore[TContext]"], StepHandler] | None) = None,
        agent_handler_factory: (Callable[["ExecutorCore[TContext]"], AgentHandler] | None) = None,
        governance_policies: tuple[GovernancePolicy, ...] | None = None,
        shadow_eval_enabled: bool | None = None,
        shadow_eval_sample_rate: float | None = None,
        shadow_eval_timeout_s: float | None = None,
        shadow_eval_judge_model: str | None = None,
        shadow_eval_sink: str | None = None,
        shadow_eval_evaluate_on_failure: bool | None = None,
    ) -> ExecutorCoreDeps[TContext]:
        serializer_obj: ISerializer = serializer or OrjsonSerializer()
        hasher_obj: IHasher = hasher or Blake3Hasher()
        cache_key_gen = cache_key_generator or DefaultCacheKeyGenerator(hasher_obj)
        backend = cache_backend or InMemoryLRUBackend(max_size=cache_size, ttl_s=cache_ttl)
        cache_manager = CacheManager(
            backend=backend,
            key_generator=cache_key_gen,
            enable_cache=enable_cache,
        )
        memory_store_obj: VectorStoreProtocol = memory_store or NullVectorStore()
        background_task_manager_obj = background_task_manager or BackgroundTaskManager()
        from .background_task_handler import BackgroundTaskHandler as _BackgroundTaskHandler

        bg_task_handler_obj = _BackgroundTaskHandler(
            None
        )  # Temporary init, updated later if needed
        # Memory manager wiring (optional, disabled by default)
        try:
            from ....embeddings import get_embedding_client, get_embedding_dimensions
        except Exception:  # pragma: no cover - optional dependency
            get_embedding_client = None  # type: ignore
            get_embedding_dimensions = None  # type: ignore

        settings = get_settings()
        sandbox_cfg = getattr(settings, "sandbox", settings)
        sandbox_mode = getattr(sandbox_cfg, "mode", getattr(settings, "sandbox_mode", "null"))
        sandbox_obj: SandboxProtocol
        if sandbox is not None:
            sandbox_obj = sandbox
        elif sandbox_mode == "remote":
            api_url = getattr(sandbox_cfg, "api_url", getattr(settings, "sandbox_api_url", None))
            if api_url:
                try:
                    sandbox_obj = RemoteSandbox(
                        api_url=api_url,
                        api_key=getattr(
                            sandbox_cfg, "api_key", getattr(settings, "sandbox_api_key", None)
                        ),
                        timeout_s=float(
                            getattr(
                                sandbox_cfg,
                                "timeout_seconds",
                                getattr(settings, "sandbox_timeout_s", 60.0),
                            )
                        ),
                        verify_ssl=bool(
                            getattr(
                                sandbox_cfg,
                                "verify_ssl",
                                getattr(settings, "sandbox_verify_ssl", True),
                            )
                        ),
                    )
                except Exception:
                    sandbox_obj = NullSandbox()
            else:
                sandbox_obj = NullSandbox()
        elif sandbox_mode == "docker":
            try:
                sandbox_obj = DockerSandbox(
                    image=getattr(sandbox_cfg, "docker_image", "python:3.13-slim"),
                    pull=bool(getattr(sandbox_cfg, "docker_pull", True)),
                    timeout_s=float(
                        getattr(
                            sandbox_cfg,
                            "timeout_seconds",
                            getattr(settings, "sandbox_timeout_s", 60.0),
                        )
                    ),
                    mem_limit=getattr(
                        sandbox_cfg,
                        "docker_mem_limit",
                        getattr(settings, "sandbox_docker_mem_limit", None),
                    ),
                    pids_limit=getattr(
                        sandbox_cfg,
                        "docker_pids_limit",
                        getattr(settings, "sandbox_docker_pids_limit", None),
                    ),
                    network_mode=getattr(
                        sandbox_cfg,
                        "docker_network_mode",
                        getattr(settings, "sandbox_docker_network_mode", None),
                    ),
                )
            except Exception:
                sandbox_obj = NullSandbox()
        else:
            sandbox_obj = NullSandbox()
        memory_enabled = bool(getattr(settings, "memory_indexing_enabled", False))
        memory_model = getattr(settings, "memory_embedding_model", None)
        memory_dimensions = getattr(settings, "memory_embedding_dimensions", None)
        if memory_dimensions is None and memory_model and get_embedding_dimensions is not None:
            memory_dimensions = get_embedding_dimensions(memory_model)
        if memory_dimensions is not None and memory_dimensions <= 0:
            raise ValueError("memory_embedding_dimensions must be a positive integer.")

        if memory_store is None and memory_enabled:
            state_uri = get_state_uri(force_reload=True)
            if state_uri:
                parsed = urlparse(state_uri)
                scheme = (parsed.scheme or "").lower()
                if scheme == "sqlite":
                    path = self._resolve_sqlite_path(parsed.path)
                    memory_store_obj = SQLiteVectorStore(str(path))
                elif scheme in {"postgres", "postgresql"}:
                    if (
                        memory_model
                        and memory_dimensions is None
                        and get_embedding_dimensions is not None
                    ):
                        raise RuntimeError(
                            "Unable to determine embedding dimensions for memory indexing. "
                            "Set FLUJO_MEMORY_EMBEDDING_DIMENSIONS or use a supported embedding model."
                        )
                    vector_dimensions = (
                        memory_dimensions or PostgresVectorStore.DEFAULT_VECTOR_DIMENSIONS
                    )
                    memory_store_obj = PostgresVectorStore(
                        state_uri, vector_dimensions=vector_dimensions
                    )

        embedder_fn = None
        if memory_enabled and memory_model and get_embedding_client is not None:
            try:
                client = get_embedding_client(memory_model)

                async def _embed(texts: list[str]) -> list[list[float]]:
                    res = await client.embed(texts)
                    return res.embeddings

                embedder_fn = _embed
            except Exception:
                embedder_fn = None
                memory_enabled = False

        memory_manager_obj: MemoryManager = (
            MemoryManager(
                store=memory_store_obj,
                embedder=embedder_fn,
                enabled=memory_enabled,
                background_task_manager=background_task_manager_obj,
            )
            if memory_enabled and embedder_fn is not None
            else NullMemoryManager()
        )

        plugin_runner_obj: IPluginRunner = plugin_runner or DefaultPluginRunner()
        agent_runner_obj: IAgentRunner = agent_runner or DefaultAgentRunner()
        validator_runner_obj: IValidatorRunner = validator_runner or DefaultValidatorRunner()

        processor_pipeline_obj: IProcessorPipeline = (
            processor_pipeline or DefaultProcessorPipeline()
        )
        usage_meter_obj: IUsageMeter = usage_meter or ThreadSafeMeter()
        telemetry_obj: ITelemetry = telemetry or DefaultTelemetry()

        timeout_runner_obj: TimeoutRunner = timeout_runner or DefaultTimeoutRunner()
        unpacker_obj: AgentResultUnpacker = unpacker or DefaultAgentResultUnpacker()
        plugin_redirector_obj: PluginRedirector = plugin_redirector or DefaultPluginRedirector(
            plugin_runner_obj, agent_runner_obj
        )
        validator_invoker_obj: ValidatorInvoker = validator_invoker or DefaultValidatorInvoker(
            validator_runner_obj
        )
        simple_step_executor_obj: SimpleStepExecutor = (
            simple_step_executor or DefaultSimpleStepExecutor()
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
            from ..step_policies import DefaultImportStepExecutor

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
        policies = governance_policies
        if policies is None:
            mode = getattr(settings, "governance_mode", "allow_all")
            custom_module = getattr(settings, "governance_policy_module", None)
            pii_scrub = bool(getattr(settings, "governance_pii_scrub", False))
            pii_strong = bool(getattr(settings, "governance_pii_strong", False))
            tool_allowlist_raw = getattr(settings, "governance_tool_allowlist", ())
            if isinstance(tool_allowlist_raw, str):
                tool_allowlist = tuple(
                    part.strip() for part in tool_allowlist_raw.split(",") if part.strip()
                )
            elif isinstance(tool_allowlist_raw, (list, tuple)):
                tool_allowlist = tuple(
                    str(item).strip() for item in tool_allowlist_raw if str(item).strip()
                )
            else:
                tool_allowlist = ()
            extras: list[GovernancePolicy] = []
            if pii_scrub:
                extras.append(PIIScrubbingPolicy(strong=pii_strong))
            if tool_allowlist:
                extras.append(
                    ToolAllowlistPolicy(allowed=frozenset(str(x) for x in tool_allowlist))
                )
            if custom_module:
                loaded = self._load_governance_policy(custom_module)
                if loaded is not None:
                    policies = tuple(extras) + (loaded,)
            if policies is None:
                if mode == "deny_all":
                    policies = (DenyAllGovernancePolicy(),)
                else:
                    policies = tuple(extras) + (AllowAllGovernancePolicy(),)
        governance_engine_obj = GovernanceEngine(policies=policies)
        shadow_eval_config = ShadowEvalConfig(
            enabled=bool(
                shadow_eval_enabled
                if shadow_eval_enabled is not None
                else getattr(settings, "shadow_eval_enabled", False)
            ),
            sample_rate=float(
                shadow_eval_sample_rate
                if shadow_eval_sample_rate is not None
                else getattr(settings, "shadow_eval_sample_rate", 0.0)
            ),
            timeout_s=float(
                shadow_eval_timeout_s
                if shadow_eval_timeout_s is not None
                else getattr(settings, "shadow_eval_timeout_s", 30.0)
            ),
            judge_model=str(
                shadow_eval_judge_model
                if shadow_eval_judge_model is not None
                else getattr(settings, "shadow_eval_judge_model", "openai:gpt-4o-mini")
            ),
            sink=str(
                shadow_eval_sink
                if shadow_eval_sink is not None
                else getattr(settings, "shadow_eval_sink", "telemetry")
            ),
            evaluate_on_failure=bool(
                shadow_eval_evaluate_on_failure
                if shadow_eval_evaluate_on_failure is not None
                else getattr(settings, "shadow_eval_evaluate_on_failure", False)
            ),
            run_level_enabled=bool(getattr(settings, "shadow_eval_run_level", False)),
        )
        shadow_evaluator_obj = ShadowEvaluator(
            config=shadow_eval_config,
            background_task_manager=background_task_manager_obj,
        )

        return ExecutorCoreDeps(
            agent_runner=agent_runner_obj,
            processor_pipeline=processor_pipeline_obj,
            validator_runner=validator_runner_obj,
            plugin_runner=plugin_runner_obj,
            usage_meter=usage_meter_obj,
            telemetry=telemetry_obj,
            quota_manager=quota_manager or QuotaManager(),
            cache_manager=cache_manager,
            memory_store=memory_store_obj,
            sandbox=sandbox_obj,
            memory_manager=memory_manager_obj,
            serializer=serializer_obj,
            hasher=hasher_obj,
            cache_key_generator=cache_key_gen,
            fallback_handler=fallback_handler or FallbackHandler(),
            hydration_manager=hydration_manager or HydrationManager(state_providers),
            background_task_manager=background_task_manager_obj,
            bg_task_handler=bg_task_handler_obj,
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
            shadow_evaluator=shadow_evaluator_obj,
        )

    @staticmethod
    def _resolve_sqlite_path(raw_path: str) -> Path:
        """Resolve sqlite path from URI path, keeping relative paths project-local."""
        if not raw_path:
            return Path("flujo_ops.db")
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate
        trimmed = raw_path.lstrip("/\\")
        return Path.cwd() / trimmed

    @staticmethod
    def _load_governance_policy(module_path: str) -> GovernancePolicy | None:
        """Dynamically load a GovernancePolicy from a module path 'pkg.mod:Class'."""
        try:
            mod_name, cls_name = module_path.split(":", 1)
        except ValueError:
            return None
        try:
            module = importlib.import_module(mod_name)
            policy_cls = getattr(module, cls_name, None)
            if policy_cls is None:
                return None
            obj = policy_cls()
            if _is_governance_policy(obj):
                return obj
            return None
        except Exception:
            return None


def _is_governance_policy(obj: object) -> TypeGuard[GovernancePolicy]:
    return callable(getattr(obj, "evaluate", None))
