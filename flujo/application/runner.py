from __future__ import annotations

import asyncio
import inspect
import weakref
import os
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
    AsyncIterator,
    Union,
    cast,
    get_type_hints,
    Literal,
)

# No direct Awaitable usage needed; avoid unused import

from pydantic import ValidationError

from ..exceptions import (
    OrchestratorError,
    PipelineContextInitializationError,
    UsageLimitExceededError,
    PipelineAbortSignal,
    ContextInheritanceError,
    InfiniteFallbackError,
    InfiniteRedirectError as _InfiniteRedirectError,
    PricingNotConfiguredError,
)
from ..domain.dsl.step import Step
from ..domain.dsl.pipeline import Pipeline
from ..domain.dsl.step import HumanInTheLoopStep
from ..domain.models import (
    BaseModel,
    PipelineResult,
    StepResult,
    UsageLimits,
    Quota,
    PipelineContext,
    HumanInteraction,
    ExecutedCommandLog,
    StepOutcome,
    Success,
    Failure,
    Paused,
    Chunk,
)
from ..domain.commands import AgentCommand
from pydantic import TypeAdapter
from ..domain.resources import AppResources
from ..domain.types import HookCallable
from ..domain.backends import ExecutionBackend
from ..infra.console_tracer import ConsoleTracer
from ..state import StateBackend, WorkflowState
from ..infra.registry import PipelineRegistry

from .core.context_manager import (
    _accepts_param,
    _extract_missing_fields,
)


from .core.hook_dispatcher import _dispatch_hook as _dispatch_hook_impl
from .core.execution_manager import ExecutionManager
from .core.state_manager import StateManager
from .core.step_coordinator import StepCoordinator

import uuid
import warnings
from ..utils.config import get_settings

_signature_cache_weak: weakref.WeakKeyDictionary[Callable[..., Any], inspect.Signature] = (
    weakref.WeakKeyDictionary()
)
_signature_cache_id: dict[int, tuple[weakref.ref[Any], inspect.Signature]] = {}
_type_hints_cache_weak: weakref.WeakKeyDictionary[Callable[..., Any], Dict[str, Any]] = (
    weakref.WeakKeyDictionary()
)
_type_hints_cache_id: dict[int, tuple[weakref.ref[Any], Dict[str, Any]]] = {}


def _cached_signature(func: Callable[..., Any]) -> inspect.Signature | None:
    """Return and cache the signature of ``func``.

    ``inspect.signature`` is relatively expensive and does not work on all
    callables. To speed up repeated calls and gracefully handle unhashable
    callables, we maintain two caches:

    - ``_signature_cache_weak`` keyed by the callable object when it is
      hashable.
    - ``_signature_cache_id`` keyed by ``id(func)`` with a weak reference to
      evict entries once the object is garbage collected.
    """
    try:
        return _signature_cache_weak[func]
    except KeyError:
        pass
    except TypeError:
        entry = _signature_cache_id.get(id(func))
        if entry is not None:
            ref, cached_sig = entry
            if ref() is func:
                return cached_sig
            if ref() is None:
                _signature_cache_id.pop(id(func), None)
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    try:
        _signature_cache_weak[func] = sig
    except TypeError:
        func_id = id(func)
        _signature_cache_id[func_id] = (
            weakref.ref(func, lambda _: _signature_cache_id.pop(func_id, None)),
            sig,
        )
    return sig


def _cached_type_hints(func: Callable[..., Any]) -> Dict[str, Any] | None:
    """Return and cache the evaluated type hints for ``func``.

    Similar to :func:`_cached_signature`, this function keeps a weak-keyed cache
    as well as an ``id``-based fallback to support unhashable callables. Any
    errors from ``get_type_hints`` are swallowed and ``None`` is returned so that
    hook dispatching can continue even for dynamically typed functions.
    """
    try:
        return _type_hints_cache_weak[func]
    except KeyError:
        pass
    except TypeError:
        entry = _type_hints_cache_id.get(id(func))
        if entry is not None:
            ref, cached = entry
            if ref() is func:
                return cached
            if ref() is None:
                _type_hints_cache_id.pop(id(func), None)
    try:
        hints = get_type_hints(func)
    except Exception:
        return None
    try:
        _type_hints_cache_weak[func] = hints
    except TypeError:
        func_id = id(func)
        _type_hints_cache_id[func_id] = (
            weakref.ref(func, lambda _: _type_hints_cache_id.pop(func_id, None)),
            hints,
        )
    return hints


_agent_command_adapter: TypeAdapter[AgentCommand] = TypeAdapter(AgentCommand)


# Alias exported for backwards compatibility
InfiniteRedirectError = _InfiniteRedirectError


RunnerInT = TypeVar("RunnerInT")
RunnerOutT = TypeVar("RunnerOutT")
ContextT = TypeVar("ContextT", bound=BaseModel)


class Flujo(Generic[RunnerInT, RunnerOutT, ContextT]):
    """Execute a pipeline sequentially.

    Parameters
    ----------
    pipeline : Pipeline | Step | None, optional
        Pipeline object to run directly. Deprecated when using ``registry``.
    registry : PipelineRegistry, optional
        Registry holding named pipelines.
    pipeline_name : str, optional
        Name of the pipeline registered in ``registry``.
    pipeline_version : str, default "latest"
        Version to load from the registry when the run starts.
    state_backend : StateBackend, optional
        Backend used to persist :class:`WorkflowState` for durable execution.
    delete_on_completion : bool, default False
        If ``True`` remove persisted state once the run finishes.
    """

    _trace_manager: Optional[Any]  # Will be TraceManager when tracing is enabled

    def __init__(
        self,
        pipeline: Pipeline[RunnerInT, RunnerOutT] | Step[RunnerInT, RunnerOutT] | None = None,
        *,
        context_model: Optional[Type[ContextT]] = None,
        initial_context_data: Optional[Dict[str, Any]] = None,
        resources: Optional[AppResources] = None,
        usage_limits: Optional[UsageLimits] = None,
        hooks: Optional[list[HookCallable]] = None,
        backend: Optional[ExecutionBackend] = None,
        state_backend: Optional[StateBackend] = None,
        delete_on_completion: bool = False,
        pipeline_version: str = "latest",
        local_tracer: Union[str, "ConsoleTracer", None] = None,
        registry: Optional[PipelineRegistry] = None,
        pipeline_name: Optional[str] = None,
        enable_tracing: bool = True,
        pipeline_id: Optional[str] = None,
    ) -> None:
        if isinstance(pipeline, Step):
            pipeline = Pipeline.from_step(pipeline)
        self.pipeline: Pipeline[RunnerInT, RunnerOutT] | None = pipeline
        self.registry = registry
        if pipeline_name is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pipeline_name = f"unnamed_{timestamp}"

            # Only warn in production environments, not in tests; and only when debug is enabled
            if (
                not get_settings().test_mode
                and not any(path in os.getcwd() for path in ["/tests/", "\\tests\\", "test_"])
                and str(os.getenv("FLUJO_DEBUG", "")).lower() in {"1", "true", "yes", "on"}
            ):
                warnings.warn(
                    "pipeline_name was not provided. Generated name based on timestamp: {}. This is discouraged for production runs.".format(
                        pipeline_name
                    ),
                    UserWarning,
                )
        self.pipeline_name = pipeline_name
        if pipeline_id is None:
            pipeline_id = str(uuid.uuid4())

            # Only warn in production environments, not in tests; and only when debug is enabled
            if (
                not get_settings().test_mode
                and not any(path in os.getcwd() for path in ["/tests/", "\\tests\\", "test_"])
                and str(os.getenv("FLUJO_DEBUG", "")).lower() in {"1", "true", "yes", "on"}
            ):
                warnings.warn(
                    "pipeline_id was not provided. Generated unique id: {}. This is discouraged for production runs.".format(
                        pipeline_id
                    ),
                    UserWarning,
                )
        self.pipeline_id = pipeline_id
        self.pipeline_version = pipeline_version
        self.context_model = context_model
        self.initial_context_data: Dict[str, Any] = initial_context_data or {}
        self.resources = resources

        # Resolve budget limits from TOML and enforce precedence/min rules (FSD-019)
        try:
            from ..infra.config_manager import ConfigManager
            from ..infra.budget_resolver import (
                resolve_limits_for_pipeline as _resolve_limits_for_pipeline,
                combine_limits as _combine_limits,
            )

            cfg = ConfigManager().load_config()
            toml_limits, _src = _resolve_limits_for_pipeline(
                getattr(cfg, "budgets", None), self.pipeline_name
            )
            # Combine with code-provided limits using most restrictive rule
            effective_limits = _combine_limits(usage_limits, toml_limits)
            self.usage_limits = effective_limits
        except Exception:
            # Defensive fallback: preserve provided limits
            self.usage_limits = usage_limits

        self.hooks: list[Any] = []
        # Tracing: honor caller's enable_tracing flag, with an env opt-out to reduce perf overhead
        # in performance/stress tests (does not affect correctness tests that assert traces).
        try:
            if str(os.getenv("FLUJO_DISABLE_TRACING", "")).strip().lower() in {
                "1",
                "true",
                "on",
                "yes",
            }:
                enable_tracing = False
        except Exception:
            pass
        # Do not auto-disable during tests by default; tests may assert presence of trace data.
        if enable_tracing:
            from flujo.tracing.manager import TraceManager, set_active_trace_manager

            self._trace_manager = TraceManager()
            # Expose the active trace manager via contextvar for processors/utilities
            try:
                set_active_trace_manager(self._trace_manager)
            except Exception:
                pass
            self.hooks.append(self._trace_manager.hook)
        else:
            self._trace_manager = None
        if hooks:
            self.hooks.extend(hooks)
        tracer_instance = None
        if isinstance(local_tracer, ConsoleTracer):
            tracer_instance = local_tracer
        elif local_tracer == "default":
            tracer_instance = ConsoleTracer()
        if tracer_instance:
            self.hooks.append(tracer_instance.hook)
        if backend is None:
            # ✅ COMPOSITION ROOT: Create and wire all dependencies
            backend = self._create_default_backend()
        self.backend = backend
        # Debug: Log backend and executor type
        from flujo.infra import telemetry

        backend_type = type(self.backend).__name__
        executor_type = getattr(getattr(self.backend, "_executor", None), "__class__", None)
        telemetry.logfire.debug(f"Flujo backend: {backend_type}, executor: {executor_type}")

        self.state_backend: StateBackend | None
        if state_backend is None:
            # Default to SQLite for durability; only use in-memory in explicit test mode
            if get_settings().test_mode:
                from ..state.backends.memory import InMemoryBackend

                self.state_backend = InMemoryBackend()
            else:
                from pathlib import Path
                from ..state.backends.sqlite import SQLiteBackend

                db_path = Path.cwd() / "flujo_ops.db"
                self.state_backend = SQLiteBackend(db_path)
        else:
            self.state_backend = state_backend
        # Track ownership to ensure we only tear down default state backends automatically.
        self._owns_state_backend = state_backend is None
        self.delete_on_completion = delete_on_completion

    def _create_default_backend(self) -> "ExecutionBackend":
        """Create a default LocalBackend with properly wired ExecutorCore.

        This method acts as the Composition Root, assembling all the
        components needed for optimal execution.
        """
        from ..application.core.executor_core import ExecutorCore
        from ..application.core.default_components import (
            OrjsonSerializer,
            Blake3Hasher,
            InMemoryLRUBackend,
            ThreadSafeMeter,
            DefaultAgentRunner,
            DefaultProcessorPipeline,
            DefaultValidatorRunner,
            DefaultPluginRunner,
            DefaultTelemetry,
        )

        # ✅ Assemble ExecutorCore with explicit high-performance components
        executor: ExecutorCore[Any] = ExecutorCore(
            serializer=OrjsonSerializer(),
            hasher=Blake3Hasher(),
            cache_backend=InMemoryLRUBackend(),
            usage_meter=ThreadSafeMeter(),
            agent_runner=DefaultAgentRunner(),
            processor_pipeline=DefaultProcessorPipeline(),
            validator_runner=DefaultValidatorRunner(),
            plugin_runner=DefaultPluginRunner(),
            telemetry=DefaultTelemetry(),
        )

        # ✅ Create LocalBackend and inject the executor
        from ..infra.backends import LocalBackend

        return LocalBackend(executor=executor)

    def disable_tracing(self) -> None:
        """Disable tracing by removing the TraceManager hook."""
        if self._trace_manager is not None:
            # Remove the TraceManager hook from the hooks list
            self.hooks = [hook for hook in self.hooks if hook != self._trace_manager.hook]
            self._trace_manager = None
        # Clear active trace manager reference
        try:
            from flujo.tracing.manager import set_active_trace_manager

            set_active_trace_manager(None)
        except Exception:
            pass

    async def aclose(self) -> None:
        """Asynchronously release runner-owned resources."""
        await self._shutdown_state_backend()

    def close(self) -> None:
        """Synchronously release runner-owned resources (best-effort in async contexts)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose())
        else:
            loop.create_task(self.aclose())

    async def __aenter__(self) -> "Flujo[RunnerInT, RunnerOutT, ContextT]":
        return self

    async def __aexit__(
        self,
        exc_type: Type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        await self.aclose()

    def __enter__(self) -> "Flujo[RunnerInT, RunnerOutT, ContextT]":
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> bool:
        self.close()
        return False

    def _ensure_pipeline(self) -> Pipeline[RunnerInT, RunnerOutT]:
        """Load the configured pipeline from the registry if needed."""
        if self.pipeline is not None:
            return self.pipeline
        if self.registry is None or self.pipeline_name is None:
            raise OrchestratorError("Pipeline not provided and registry missing")
        if self.pipeline_version == "latest":
            version = self.registry.get_latest_version(self.pipeline_name)
            if version is None:
                raise OrchestratorError(f"No pipeline registered under name '{self.pipeline_name}'")
            self.pipeline_version = version
            pipe = self.registry.get(self.pipeline_name, version)
        else:
            pipe = self.registry.get(self.pipeline_name, self.pipeline_version)
        if pipe is None:
            raise OrchestratorError(
                f"Pipeline '{self.pipeline_name}' version '{self.pipeline_version}' not found"
            )
        self.pipeline = pipe
        return pipe

    async def _dispatch_hook(
        self,
        event_name: Literal[
            "pre_run",
            "post_run",
            "pre_step",
            "post_step",
            "on_step_failure",
        ],
        **kwargs: Any,
    ) -> None:
        """Invoke registered hooks for ``event_name``."""

        await _dispatch_hook_impl(self.hooks, event_name, **kwargs)

    async def _execute_steps(
        self,
        start_idx: int,
        data: Any,
        context: Optional[ContextT],
        result: PipelineResult[ContextT],
        *,
        stream_last: bool = False,
        run_id: str | None = None,
        state_backend: StateBackend | None = None,
        state_created_at: datetime | None = None,
    ) -> AsyncIterator[Any]:
        """Execute pipeline steps using the new execution manager.

        This method now delegates to the ExecutionManager which coordinates
        all execution components in a clean, testable way.
        """
        assert self.pipeline is not None

        # Create execution manager with all components
        state_manager: StateManager[ContextT] = StateManager[ContextT](state_backend)
        step_coordinator: StepCoordinator[ContextT] = StepCoordinator[ContextT](
            self.hooks, self.resources
        )

        # Build root quota if usage limits are defined
        root_quota = None
        if self.usage_limits is not None:
            total_cost_limit = (
                float(self.usage_limits.total_cost_usd_limit)
                if self.usage_limits.total_cost_usd_limit is not None
                else float("inf")
            )
            total_tokens_limit = int(self.usage_limits.total_tokens_limit or 0)
            root_quota = Quota(total_cost_limit, total_tokens_limit)

        execution_manager = ExecutionManager(
            self.pipeline,
            backend=self.backend,  # ✅ Pass the backend to the execution manager.
            state_manager=state_manager,
            usage_limits=self.usage_limits,
            step_coordinator=step_coordinator,
            root_quota=root_quota,
        )

        # Execute steps using the manager
        async for item in execution_manager.execute_steps(
            start_idx=start_idx,
            data=data,
            context=context,
            result=result,
            stream_last=stream_last,
            run_id=run_id,
            state_created_at=state_created_at,
        ):
            yield item

    async def run_async(
        self,
        initial_input: RunnerInT,
        *,
        run_id: str | None = None,
        initial_context_data: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[PipelineResult[ContextT]]:
        """Run the pipeline asynchronously.

        Parameters
        ----------
        run_id:
            Optional identifier for this run. When provided the runner will load
            and persist state under this ID, enabling durable execution without
            embedding the ID in the context model.

        This method should be used when an asyncio event loop is already
        running, such as within Jupyter notebooks or async web frameworks.

        It yields any streaming output from the final step and then the final
        ``PipelineResult`` object.
        """
        async def _run_generator() -> AsyncIterator[PipelineResult[ContextT]]:

            # Dev-only deprecation: warn when legacy runner is used and flag is enabled
            try:
                if get_settings().warn_legacy:
                    warnings.warn(
                        "Legacy runner path (run_async yielding PipelineResult) in use; prefer run_outcomes_async.",
                        DeprecationWarning,
                    )
            except Exception:
                pass

            # Debug: log provided initial_context_data for visibility in map-over tests
            try:
                from flujo.infra import telemetry

                telemetry.logfire.debug(
                    f"Runner.run_async received initial_context_data keys={list(initial_context_data.keys()) if isinstance(initial_context_data, dict) else None}"
                )
            except Exception:
                pass
            current_context_instance: Optional[ContextT] = None
            if self.context_model is not None:
                try:
                    # CRITICAL FIX: Create a deep copy of initial context data to prevent shared state
                    import copy

                    context_data = (
                        copy.deepcopy(self.initial_context_data) if self.initial_context_data else {}
                    )
                    if initial_context_data:
                        # Also deep copy the initial_context_data to prevent shared state
                        context_data.update(copy.deepcopy(initial_context_data))
                    if run_id is not None:
                        context_data["run_id"] = run_id

                    # Process context_data to reconstruct custom types using the deserializer registry
                    from flujo.utils.serialization import lookup_custom_deserializer

                    processed_context_data = {}
                    for key, value in context_data.items():
                        if key in self.context_model.model_fields:
                            field_info = self.context_model.model_fields[key]
                            field_type = field_info.annotation
                            # Try to reconstruct custom types using the global deserializer registry
                            if field_type is not None and isinstance(value, dict):
                                custom_deserializer = lookup_custom_deserializer(field_type)
                                if custom_deserializer:
                                    try:
                                        reconstructed_value = custom_deserializer(value)
                                        processed_context_data[key] = reconstructed_value
                                        continue
                                    except Exception:
                                        pass  # Fallback to original value
                            processed_context_data[key] = value
                        else:
                            processed_context_data[key] = value

                    try:
                        telemetry.logfire.info(
                            f"Runner.run_async building context with data: {processed_context_data}"
                        )
                    except Exception:
                        pass
                    current_context_instance = self.context_model(**processed_context_data)
                    try:
                        telemetry.logfire.debug(
                            f"Runner.run_async created context instance of type: {self.context_model.__name__}"
                        )
                    except Exception:
                        pass
                except ValidationError as e:
                    telemetry.logfire.error(
                        f"Context initialization failed for model {self.context_model.__name__}: {e}"
                    )
                    msg = f"Failed to initialize context with model {self.context_model.__name__} and initial data."
                    if any(err.get("loc") == ("initial_prompt",) for err in e.errors()):
                        msg += " `initial_prompt` field required. Your custom context model must inherit from flujo.domain.models.PipelineContext."
                    msg += f" Validation errors:\n{e}"
                    raise PipelineContextInitializationError(msg) from e

            else:
                # When no custom context model is provided, use default PipelineContext and
                # merge runner.initial_context_data and provided initial_context_data for templating support.
                current_context_instance = cast(
                    Optional[ContextT], PipelineContext(initial_prompt=str(initial_input))
                )
                try:
                    import copy as _copy

                    merged_data: Dict[str, Any] = {}
                    if isinstance(self.initial_context_data, dict):
                        merged_data.update(_copy.deepcopy(self.initial_context_data))
                    if isinstance(initial_context_data, dict):
                        merged_data.update(_copy.deepcopy(initial_context_data))
                    for key, value in merged_data.items():
                        if key == "scratchpad" and isinstance(value, dict):
                            # Merge scratchpad dictionaries
                            try:
                                scratch = getattr(current_context_instance, "scratchpad", None)
                                if isinstance(scratch, dict):
                                    scratch.update(value)
                            except Exception:
                                pass
                        elif key in ("initial_prompt", "run_id"):
                            # Respect explicit initial_prompt/run_id if provided
                            object.__setattr__(current_context_instance, key, value)
                        else:
                            # Expose custom keys as attributes for easy access in templates (e.g., context.customer_first_question)
                            object.__setattr__(current_context_instance, key, value)
                except Exception:
                    # Non-fatal: fallback to plain context
                    pass
                if run_id is not None:
                    object.__setattr__(current_context_instance, "run_id", run_id)

            # Initialize _artifacts for refine_until functionality
            if hasattr(current_context_instance, "__dict__"):
                if not hasattr(current_context_instance, "_artifacts"):
                    object.__setattr__(current_context_instance, "_artifacts", [])

            if isinstance(current_context_instance, PipelineContext):
                current_context_instance.scratchpad["status"] = "running"

            data: Any = initial_input
            pipeline_result_obj: PipelineResult[ContextT] = PipelineResult()
            start_idx = 0
            state_created_at: datetime | None = None
            # Initialize state manager and load existing state if available
            state_manager: StateManager[ContextT] = StateManager[ContextT](self.state_backend)
            # Determine run_id early for tracing; prefer explicit run_id, else context.run_id if present
            run_id_for_state = run_id or getattr(current_context_instance, "run_id", None)

            # If we didn't get an explicit run_id, consult the state manager
            # to support durable resume semantics
            if run_id_for_state is None:
                run_id_for_state = state_manager.get_run_id_from_context(current_context_instance)

            if run_id_for_state:
                (
                    context,
                    last_output,
                    current_idx,
                    created_at,
                    pipeline_name,
                    pipeline_version,
                    step_history,
                ) = await state_manager.load_workflow_state(run_id_for_state, self.context_model)
                if context is not None:
                    # Resume from persisted state
                    current_context_instance = context
                    start_idx = current_idx
                    state_created_at = created_at
                    if start_idx > 0:
                        data = last_output
                        # Restore step history from persisted state
                        pipeline_result_obj.step_history = step_history

                    # Restore pipeline version from state
                    if pipeline_version is not None:
                        self.pipeline_version = pipeline_version
                    if pipeline_name is not None:
                        self.pipeline_name = pipeline_name

                    # Ensure pipeline is loaded with correct version
                    self._ensure_pipeline()

                    # Validate step index
                    assert self.pipeline is not None
                    if start_idx > len(self.pipeline.steps):
                        raise OrchestratorError(
                            f"Invalid persisted step index {start_idx} for pipeline with {len(self.pipeline.steps)} steps"
                        )
                else:
                    # New run, record start metadata
                    self._ensure_pipeline()
                    now = datetime.now(timezone.utc).isoformat()
                    await state_manager.record_run_start(
                        run_id_for_state,
                        self.pipeline_id,
                        self.pipeline_name or "unknown",
                        self.pipeline_version,
                        created_at=now,
                        updated_at=now,
                    )

                # Persist initial state with optimized path only for multi-step pipelines
                # to reduce overhead on simple single-step runs used in micro-benchmarks.
                try:
                    self._ensure_pipeline()
                    _num_steps = len(self.pipeline.steps) if self.pipeline is not None else 0
                except Exception:
                    _num_steps = 0
                if _num_steps > 1:
                    await state_manager.persist_workflow_state_optimized(
                        run_id=run_id_for_state,
                        context=current_context_instance,
                        current_step_index=start_idx,
                        last_step_output=data,
                        status="running",
                        state_created_at=state_created_at,
                    )
            else:
                self._ensure_pipeline()
            cancelled = False
            paused = False
            try:
                await self._dispatch_hook(
                    "pre_run",
                    initial_input=initial_input,
                    context=current_context_instance,
                    resources=self.resources,
                    run_id=run_id_for_state,
                    pipeline_name=self.pipeline_name,
                    pipeline_version=self.pipeline_version,
                    initial_budget_cost_usd=(
                        float(self.usage_limits.total_cost_usd_limit)
                        if self.usage_limits and self.usage_limits.total_cost_usd_limit is not None
                        else None
                    ),
                    initial_budget_tokens=(
                        int(self.usage_limits.total_tokens_limit)
                        if self.usage_limits and self.usage_limits.total_tokens_limit is not None
                        else None
                    ),
                )
                _yielded_pipeline_result = False
                async for chunk in self._execute_steps(
                    start_idx,
                    data,
                    cast(Optional[ContextT], current_context_instance),
                    pipeline_result_obj,
                    stream_last=True,
                    run_id=run_id_for_state,
                    state_backend=self.state_backend,
                    state_created_at=state_created_at,
                ):
                    # Track if the manager yielded a PipelineResult (e.g., paused/failed early)
                    # Debug: verify isinstance behavior locally
                    # Remove or comment out in commits if noisy
                    # BEGIN DEBUG
                    # print(f"ISINSTANCE? {isinstance(chunk, PipelineResult)} class={type(chunk)}")
                    # END DEBUG
                    if isinstance(chunk, PipelineResult):
                        # print("RUNNER saw PipelineResult from manager")
                        # Finalization safety: only pad when a missing-outcome was detected.
                        # This avoids padding when steps were legitimately skipped (e.g., conditionals).
                        try:
                            expected = len(self.pipeline.steps) if self.pipeline is not None else 0
                        except Exception:
                            expected = 0
                        have = len(chunk.step_history)
                        if expected > 0 and have < expected and start_idx == 0:
                            # Pad only when we suspect a missing-outcome scenario:
                            # - explicit 'no terminal outcome' feedback, OR
                            # - no recorded failures so far (all successes) but history is short
                            missing_outcome_detected = any(
                                isinstance(getattr(sr, "feedback", None), str)
                                and "no terminal outcome" in sr.feedback.lower()
                                for sr in chunk.step_history
                            ) or all(getattr(sr, "success", True) for sr in chunk.step_history)
                            if missing_outcome_detected:
                                from flujo.domain.models import StepResult as _SR

                                for j in range(have, expected):
                                    try:
                                        missing_name = getattr(
                                            self.pipeline.steps[j], "name", f"step_{j}"
                                        )
                                    except Exception:
                                        missing_name = f"step_{j}"
                                    synthesized = _SR(
                                        name=str(missing_name),
                                        success=False,
                                        output=None,
                                        attempts=0,
                                        latency_s=0.0,
                                        token_counts=0,
                                        cost_usd=0.0,
                                        feedback="Agent produced no terminal outcome",
                                        branch_context=None,
                                        metadata_={},
                                        step_history=[],
                                    )
                                    StepCoordinator().update_pipeline_result(chunk, synthesized)
                        pipeline_result_obj = chunk
                        _yielded_pipeline_result = True
                    # print(f"RUNNER yield {type(chunk).__name__}")
                    yield chunk
                # After streaming, yield the final PipelineResult for sync runners
                if not _yielded_pipeline_result:
                    # Finalization safety fallback: only pad when a missing-outcome was
                    # detected in the current history to avoid padding legitimate skips.
                    try:
                        expected = len(self.pipeline.steps) if self.pipeline is not None else 0
                    except Exception:
                        expected = 0
                    have = len(pipeline_result_obj.step_history)
                    if expected > 0 and have < expected and start_idx == 0:
                        missing_outcome_detected = any(
                            isinstance(getattr(sr, "feedback", None), str)
                            and "no terminal outcome" in sr.feedback.lower()
                            for sr in pipeline_result_obj.step_history
                        ) or all(
                            getattr(sr, "success", True) for sr in pipeline_result_obj.step_history
                        )
                        if missing_outcome_detected:
                            from flujo.domain.models import StepResult as _SR

                            for j in range(have, expected):
                                try:
                                    missing_name = getattr(self.pipeline.steps[j], "name", f"step_{j}")
                                except Exception:
                                    missing_name = f"step_{j}"
                                synthesized = _SR(
                                    name=str(missing_name),
                                    success=False,
                                    output=None,
                                    attempts=0,
                                    latency_s=0.0,
                                    token_counts=0,
                                    cost_usd=0.0,
                                    feedback="Agent produced no terminal outcome",
                                    branch_context=None,
                                    metadata_={},
                                    step_history=[],
                                )
                                StepCoordinator().update_pipeline_result(
                                    pipeline_result_obj, synthesized
                                )
                    # print("RUNNER yield final PipelineResult")
                    yield pipeline_result_obj
            except asyncio.CancelledError:
                telemetry.logfire.info("Pipeline cancelled")
                cancelled = True
                # Ensure we don't try to persist state during cancellation
                try:
                    yield pipeline_result_obj
                except Exception:
                    # Ignore any errors during yield on cancellation
                    pass
                return
            except PipelineAbortSignal as e:
                telemetry.logfire.debug(str(e))
                paused = True
            except (UsageLimitExceededError, PricingNotConfiguredError) as e:
                if current_context_instance is not None:
                    assert self.pipeline is not None
                    execution_manager: ExecutionManager[ContextT] = ExecutionManager[ContextT](
                        self.pipeline
                    )
                    execution_manager.set_final_context(
                        pipeline_result_obj,
                        cast(Optional[ContextT], current_context_instance),
                    )
                    # Update the UsageLimitExceededError result with the current pipeline step_history
                    if isinstance(e, UsageLimitExceededError):
                        if e.result is None:
                            e.result = pipeline_result_obj
                        else:
                            # Preserve the exception's step_history if it's more complete
                            if len(e.result.step_history) > len(pipeline_result_obj.step_history):
                                pipeline_result_obj.step_history = e.result.step_history
                            else:
                                e.result.step_history = pipeline_result_obj.step_history
                raise
            finally:
                if (
                    self._trace_manager is not None
                    and getattr(self._trace_manager, "_root_span", None) is not None
                ):
                    pipeline_result_obj.trace_tree = self._trace_manager._root_span
                # If we resumed from a paused state with human input, emit a resumed event on the last step span if possible
                if current_context_instance is not None:
                    assert self.pipeline is not None
                    # Persist final state using ExecutionManager with the state_manager from this run
                    exec_manager = ExecutionManager(
                        self.pipeline,
                        state_manager=state_manager,
                    )
                    # set_final_context expects the same context generic; controlled cast is safe here
                    # set_final_context expects ContextT; controlled cast from PipelineContext
                    exec_manager.set_final_context(
                        pipeline_result_obj,
                        cast(Optional[ContextT], current_context_instance),
                    )
                    # Initialize final_status default for cases with no step history
                    final_status: Literal["running", "paused", "completed", "failed", "cancelled"] = (
                        "failed"
                    )
                    if cancelled:
                        final_status = "failed"
                    elif paused or (
                        isinstance(current_context_instance, PipelineContext)
                        and current_context_instance.scratchpad.get("status") == "paused"
                    ):
                        final_status = "paused"
                    elif pipeline_result_obj.step_history:
                        # Completion gate with resume-awareness
                        try:
                            expected = len(self.pipeline.steps) if self.pipeline is not None else None
                        except Exception:
                            expected = None
                        executed_success = all(s.success for s in pipeline_result_obj.step_history)
                        if start_idx > 0:
                            # Resumed run: consider success of executed tail as completion
                            final_status = "completed" if executed_success else "failed"
                        elif (
                            expected is not None
                            and len(pipeline_result_obj.step_history) == expected
                            and executed_success
                        ):
                            final_status = "completed"
                        else:
                            final_status = "failed"
                    else:
                        # Zero-step pipelines: treat as completed
                        try:
                            num_steps = len(self.pipeline.steps) if self.pipeline is not None else 0
                        except Exception:
                            num_steps = 0
                        if num_steps == 0:
                            final_status = "completed"
                    # Do not synthesize placeholders at the runner level; ExecutionManager
                    # already pads histories for missing-outcome scenarios. Runner padding
                    # breaks pause/resume and early-abort semantics.

                    await exec_manager.persist_final_state(
                        run_id=run_id_for_state,
                        context=current_context_instance,
                        result=pipeline_result_obj,
                        start_idx=start_idx,
                        state_created_at=state_created_at,
                        final_status=final_status,
                    )
                    # Reflect final status on PipelineResult.success for back-compat
                    try:
                        pipeline_result_obj.success = final_status == "completed"
                    except Exception:
                        pass
                    # Async cleanup: delete persisted workflow state if requested
                    if (
                        self.delete_on_completion
                        and final_status == "completed"
                        and run_id_for_state is not None
                    ):
                        # Remove via StateManager
                        await state_manager.delete_workflow_state(run_id_for_state)
                        # Remove via backend if available
                        try:
                            if self.state_backend is not None:
                                await self.state_backend.delete_state(run_id_for_state)
                        except Exception:
                            pass
                        # Fallback: clear backend store attribute
                        try:
                            if self.state_backend is not None:
                                store = getattr(self.state_backend, "_store", None)
                                if isinstance(store, dict):
                                    store.clear()
                        except Exception:
                            pass
                try:
                    await self._dispatch_hook(
                        "post_run",
                        pipeline_result=pipeline_result_obj,
                        context=current_context_instance,
                        resources=self.resources,
                    )
                except asyncio.CancelledError:
                    # Don't execute hooks if we're being cancelled
                    telemetry.logfire.info("Skipping post_run hook due to cancellation")
                except PipelineAbortSignal as e:
                    # Quiet by default; only surface on --debug
                    telemetry.logfire.debug(str(e))

            yield pipeline_result_obj
            return
        try:
            async for item in _run_generator():
                yield item
        finally:
            await self._shutdown_state_backend()

    async def run_outcomes_async(
        self,
        initial_input: RunnerInT,
        *,
        run_id: str | None = None,
        initial_context_data: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[StepOutcome[StepResult]]:
        """Run the pipeline and yield typed StepOutcome events.

        - Non-streaming: yields a single Success containing the final StepResult
        - Streaming: yields zero or more Chunk items and a final Success
        - Failure: yields Failure outcome immediately when a step fails
        - Pause: yields Paused when a HITL step pauses execution
        """
        # Execute underlying steps, translating legacy values to StepOutcome
        pipeline_result_obj: PipelineResult[ContextT] = PipelineResult()
        last_step_result: StepResult | None = None
        try:
            async for item in self.run_async(
                initial_input, run_id=run_id, initial_context_data=initial_context_data
            ):
                if isinstance(item, StepOutcome):
                    if isinstance(item, Success):
                        last_step_result = item.step_result
                    yield item
                elif isinstance(item, StepResult):
                    last_step_result = item
                    if item.success:
                        yield Success(step_result=item)
                    else:
                        yield Failure(
                            error=Exception(item.feedback or "step failed"),
                            feedback=item.feedback,
                            step_result=item,
                        )
                elif isinstance(item, PipelineResult):
                    pipeline_result_obj = item
                else:
                    # Streaming chunk (legacy); wrap into Chunk outcome
                    yield Chunk(data=item)
        except PipelineAbortSignal:
            # Try to extract pause message from context if present
            try:
                ctx = pipeline_result_obj.final_pipeline_context
                msg = None
                if isinstance(ctx, PipelineContext):
                    msg = ctx.scratchpad.get("pause_message")
            except Exception:
                msg = None
            yield Paused(message=msg or "Paused for HITL")
            return

        # If manager swallowed the abort into a PipelineResult with paused context, emit Paused
        try:
            if isinstance(pipeline_result_obj, PipelineResult):
                ctx = pipeline_result_obj.final_pipeline_context
                if isinstance(ctx, PipelineContext):
                    status = ctx.scratchpad.get("status") if hasattr(ctx, "scratchpad") else None
                    if status == "paused":
                        msg = ctx.scratchpad.get("pause_message")
                        yield Paused(message=msg or "Paused for HITL")
                        return
        except Exception:
            pass

        # Emit final Success/Failure outcome based on the last step result
        if pipeline_result_obj.step_history:
            last = pipeline_result_obj.step_history[-1]
            if last.success:
                yield Success(step_result=last)
            else:
                yield Failure(
                    error=Exception(last.feedback or "step failed"),
                    feedback=last.feedback,
                    step_result=last,
                )
        elif last_step_result is not None:
            if last_step_result.success:
                yield Success(step_result=last_step_result)
            else:
                yield Failure(
                    error=Exception(last_step_result.feedback or "step failed"),
                    feedback=last_step_result.feedback,
                    step_result=last_step_result,
                )
        else:
            # If paused context exists, emit Paused; otherwise synthesize minimal success
            try:
                ctx = pipeline_result_obj.final_pipeline_context
                if isinstance(ctx, PipelineContext):
                    status = ctx.scratchpad.get("status") if hasattr(ctx, "scratchpad") else None
                    if status == "paused":
                        msg = ctx.scratchpad.get("pause_message")
                        yield Paused(message=msg or "Paused for HITL")
                        return
            except Exception:
                pass
            yield Success(step_result=StepResult(name="<no-steps>", success=True))

    async def stream_async(
        self,
        initial_input: RunnerInT,
        *,
        initial_context_data: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Any]:
        # Determine if the pipeline supports streaming by checking the last step's agent
        pipeline = self._ensure_pipeline()
        last_step = pipeline.steps[-1]
        has_stream = hasattr(last_step.agent, "stream")
        if not has_stream:
            # Non-streaming pipeline: yield only the final PipelineResult
            final_result: PipelineResult[ContextT] | None = None
            async for item in self.run_async(
                initial_input, initial_context_data=initial_context_data
            ):
                final_result = item
            if final_result is not None:
                yield final_result
        else:
            # Streaming pipeline: unwrap typed Chunk outcomes into raw chunks for legacy contract
            # and suppress intermediate Success/Failure outcomes — callers expect only
            # raw chunks and the final PipelineResult.
            seen_chunks: set[str] = set()
            async for item in self.run_async(
                initial_input, initial_context_data=initial_context_data
            ):
                from ..domain.models import Chunk as _Chunk
                from ..domain.models import StepOutcome as _StepOutcome

                if isinstance(item, _Chunk):
                    # Suppress duplicate streaming chunks across layers if they repeat
                    try:
                        k = str(item.data)
                        if k in seen_chunks:
                            continue
                        seen_chunks.add(k)
                    except Exception:
                        pass
                    yield item.data
                elif isinstance(item, _StepOutcome):
                    # Drop Success/Failure/Paused outcomes from streaming surface
                    continue
                else:
                    yield item

    def run(
        self,
        initial_input: RunnerInT,
        *,
        run_id: str | None = None,
        initial_context_data: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult[ContextT]:
        """Run the pipeline synchronously.

        This helper should only be called from code that is not already running
        inside an asyncio event loop.  If a running loop is detected a
        ``TypeError`` is raised instructing the user to use ``run_async``
        instead.
        """
        try:
            asyncio.get_running_loop()
            raise TypeError(
                "Flujo.run() cannot be called from a running event loop. "
                "If you are in an async environment (like Jupyter, FastAPI, or an "
                "`async def` function), you must use the `run_async()` method."
            )
        except RuntimeError:
            # No loop running, safe to proceed
            pass

        async def _consume() -> PipelineResult[ContextT]:
            result: PipelineResult[ContextT] | None = None
            async for r in self.run_async(
                initial_input,
                run_id=run_id,
                initial_context_data=initial_context_data,
            ):
                result = r
            # Attach trace tree to result before returning
            if (
                result is not None
                and self._trace_manager is not None
                and getattr(self._trace_manager, "_root_span", None) is not None
            ):
                result.trace_tree = self._trace_manager._root_span
            assert result is not None
            return result

        return asyncio.run(_consume())

    async def resume_async(
        self, paused_result: PipelineResult[ContextT], human_input: Any
    ) -> PipelineResult[ContextT]:
        """Resume a paused pipeline with human input."""
        ctx: ContextT | None = paused_result.final_pipeline_context
        # The ``scratchpad`` on the context stores bookkeeping information about
        # paused pipelines.  If the context is missing or the status flag is not
        # ``"paused"`` we cannot safely resume.
        if ctx is None:
            raise OrchestratorError("Cannot resume pipeline without context")
        scratch = getattr(ctx, "scratchpad", {})
        if scratch.get("status") != "paused":
            raise OrchestratorError("Pipeline is not paused")
        self._ensure_pipeline()
        assert self.pipeline is not None
        start_idx = len(paused_result.step_history)
        if start_idx >= len(self.pipeline.steps):
            raise OrchestratorError("No steps remaining to resume")
        paused_step = self.pipeline.steps[start_idx]

        if isinstance(paused_step, HumanInTheLoopStep) and paused_step.input_schema is not None:
            human_input = paused_step.input_schema.model_validate(human_input)

        if isinstance(ctx, PipelineContext):
            ctx.hitl_history.append(
                HumanInteraction(
                    message_to_human=scratch.get("pause_message", ""),
                    human_response=human_input,
                )
            )
            ctx.scratchpad["status"] = "running"

        # Ensure conversational history captures the user's resume input when present.
        # This complements loop policy handling, which may not see the HITL step in
        # step_history on resume (since we inject a synthetic paused_step_result).
        try:
            if isinstance(ctx, PipelineContext):
                # Ensure list container exists
                if not isinstance(getattr(ctx, "conversation_history", None), list):
                    setattr(ctx, "conversation_history", [])
                from flujo.domain.models import ConversationTurn, ConversationRole

                # Append user turn if not a duplicate of the last entry
                hist = ctx.conversation_history
                last_content = hist[-1].content if hist else None
                text = str(human_input)
                if text and text != last_content:
                    hist.append(ConversationTurn(role=ConversationRole.user, content=text))
        except Exception:
            pass
        paused_step_result = StepResult(
            name=paused_step.name,
            output=human_input,
            success=True,
            attempts=1,
        )

        # Apply sink_to to context
        # NOTE: This is applied here (in runner) rather than in the HITL executor
        # because the executor works with forked contexts (in conditionals/loops),
        # while the runner has access to the main context that gets persisted.
        if hasattr(paused_step, "sink_to") and paused_step.sink_to and ctx is not None:
            try:
                from flujo.utils.context import set_nested_context_field

                set_nested_context_field(ctx, paused_step.sink_to, human_input)
            except Exception:
                # Sink failure is graceful - just skip it
                pass

        # Ensure the last HITL output is accessible to subsequent steps via
        # context.scratchpad['steps'][<step_name>] so templates like
        # {{ steps.ask_user_for_clarification.output }} resolve after resume.
        try:
            if isinstance(ctx, PipelineContext):
                sp = getattr(ctx, "scratchpad", None)
                if not isinstance(sp, dict):
                    setattr(ctx, "scratchpad", {"steps": {}})
                    sp = getattr(ctx, "scratchpad", None)
                if isinstance(sp, dict):
                    steps_map = sp.get("steps")
                    if not isinstance(steps_map, dict):
                        steps_map = {}
                        sp["steps"] = steps_map
                    # Compact snapshot: stringify and cap size
                    try:
                        val = human_input
                        if isinstance(val, bytes):
                            try:
                                val = val.decode("utf-8", errors="ignore")
                            except Exception:
                                val = str(val)
                        else:
                            val = str(val)
                        if len(val) > 1024:
                            val = val[:1024]
                        steps_map[getattr(paused_step, "name", "")] = val
                    except Exception:
                        steps_map[getattr(paused_step, "name", "")] = ""
        except Exception:
            pass
        if isinstance(ctx, PipelineContext):
            pending = ctx.scratchpad.pop("paused_step_input", None)
            if pending is not None:
                # If we already have a concrete AgentCommand instance, use it directly
                try:
                    from flujo.domain.commands import (
                        RunAgentCommand as _Run,
                        AskHumanCommand as _Ask,
                        FinishCommand as _Fin,
                    )

                    if isinstance(pending, (_Run, _Ask, _Fin)):
                        pending_cmd = pending
                    else:
                        pending_cmd = _agent_command_adapter.validate_python(pending)
                except ValidationError:
                    pending_cmd = None
                except Exception:
                    pending_cmd = None
                if pending_cmd is not None:
                    log_entry = ExecutedCommandLog(
                        turn=len(ctx.command_log) + 1,
                        generated_command=pending_cmd,
                        execution_result=human_input,
                    )
                    ctx.command_log.append(log_entry)
                    try:
                        if isinstance(ctx.scratchpad, dict):
                            ctx.scratchpad["loop_last_output"] = log_entry
                    except Exception:
                        pass
                else:
                    # If we cannot reconstruct the command, still record an AskHuman with the pause message
                    try:
                        from flujo.domain.commands import AskHumanCommand as _Ask

                        log_entry = ExecutedCommandLog(
                            turn=len(ctx.command_log) + 1,
                            generated_command=_Ask(question=scratch.get("pause_message", "Paused")),
                            execution_result=human_input,
                        )
                        ctx.command_log.append(log_entry)
                        try:
                            if isinstance(ctx.scratchpad, dict):
                                ctx.scratchpad["loop_last_output"] = log_entry
                        except Exception:
                            pass
                    except Exception:
                        pass
        # Resume semantics:
        # - If the paused step is an explicit HumanInTheLoopStep, mark it successful and continue to next.
        # - Otherwise (e.g., loop or generic agent step that paused internally), re-run the same step
        #   with the human_input as data so the step can consume it appropriately.
        from ..domain.dsl.step import HumanInTheLoopStep as _HITL

        try:
            if self._trace_manager is not None:
                summary = str(human_input)
                if isinstance(summary, str) and len(summary) > 500:
                    summary = summary[:500] + "..."
                self._trace_manager.add_event("flujo.resumed", {"human_input": summary})
        except Exception:
            pass

        if isinstance(paused_step, _HITL):
            # Replace the prior paused-failure record for this HITL step, if present
            if paused_result.step_history:
                last = paused_result.step_history[-1]
                if last.name == paused_step.name and not last.success:
                    paused_result.step_history[-1] = paused_step_result
                else:
                    paused_result.step_history.append(paused_step_result)
            else:
                paused_result.step_history.append(paused_step_result)
            # If the paused step updates context, merge the human_input output into context before next step
            try:
                if getattr(paused_step, "updates_context", False) and isinstance(
                    ctx, PipelineContext
                ):
                    from .core.context_adapter import _build_context_update, _inject_context

                    update_data = _build_context_update(human_input)
                    if update_data:
                        validation_error = _inject_context(ctx, update_data, type(ctx))
                        if validation_error:
                            raise OrchestratorError(
                                f"Failed to merge human input into context: {validation_error}"
                            )
            except Exception as _merge_err:
                # Defensive: log but continue; downstream may still succeed without context merge
                try:
                    from ..infra import telemetry as _telemetry

                    _telemetry.logfire.warning(
                        f"Resume context merge warning for step '{paused_step.name}': {_merge_err}"
                    )
                except Exception:
                    pass
            # Finalize the lingering HITL step span so subsequent steps are siblings, not children.
            # This mirrors normal success flow and ensures TraceManager pops the pre_step span.
            try:
                await self._dispatch_hook(
                    "post_step",
                    step_result=paused_step_result,
                    context=ctx,
                    resources=self.resources,
                )
            except Exception:
                # Never let tracing break resume semantics
                pass
            data = human_input
            resume_start_idx = start_idx + 1
        else:
            # Do not append synthetic success for non-HITL steps; allow the step to handle input
            data = human_input
            resume_start_idx = start_idx

        run_id_for_state = getattr(ctx, "run_id", None)
        state_created_at: datetime | None = None
        if self.state_backend is not None and run_id_for_state is not None:
            loaded = await self.state_backend.load_state(run_id_for_state)
            if loaded is not None:
                wf_state_loaded = WorkflowState.model_validate(loaded)
                state_created_at = wf_state_loaded.created_at
        from ..exceptions import PipelineAbortSignal as _Abort

        try:
            async for _ in self._execute_steps(
                resume_start_idx,
                data,
                cast(Optional[ContextT], ctx),
                paused_result,
                stream_last=False,
                run_id=run_id_for_state,
                state_backend=self.state_backend,
                state_created_at=state_created_at,
            ):
                pass
        except _Abort:
            # Swallow pause during resume and return the partial result as paused
            if isinstance(ctx, PipelineContext):
                ctx.scratchpad["status"] = "paused"

        final_status: Literal[
            "running",
            "paused",
            "completed",
            "failed",
            "cancelled",
        ]
        if paused_result.step_history:
            final_status = (
                "completed" if all(s.success for s in paused_result.step_history) else "failed"
            )
        else:
            final_status = "failed"
        if isinstance(ctx, PipelineContext):
            if ctx.scratchpad.get("status") == "paused":
                final_status = "paused"
            ctx.scratchpad["status"] = final_status

        # Use execution manager to persist final state
        state_manager: StateManager[ContextT] = StateManager[ContextT](self.state_backend)
        assert self.pipeline is not None
        execution_manager: ExecutionManager[ContextT] = ExecutionManager[ContextT](
            self.pipeline,
            state_manager=state_manager,
        )
        await execution_manager.persist_final_state(
            run_id=run_id_for_state,
            context=ctx,
            result=paused_result,
            start_idx=len(paused_result.step_history),
            state_created_at=state_created_at,
            final_status=final_status,
        )

        # Reflect final status on PipelineResult.success for back-compat
        try:
            paused_result.success = final_status == "completed"
        except Exception:
            pass

        # Delete state if delete_on_completion is True and pipeline completed successfully
        if (
            self.delete_on_completion
            and final_status == "completed"
            and run_id_for_state is not None
        ):
            # Remove persisted workflow state via StateManager
            await state_manager.delete_workflow_state(run_id_for_state)
            # Explicitly delete raw state entry from backend to ensure cleanup
            try:
                if self.state_backend is not None:
                    await self.state_backend.delete_state(run_id_for_state)
            except Exception:
                # Ignore errors during deletion to avoid breaking flow
                pass
            # Final fallback: completely clear backend store to remove any residual state
            try:
                if self.state_backend is not None:
                    store = getattr(self.state_backend, "_store", None)
                    if isinstance(store, dict):
                        store.clear()
            except Exception:
                pass

        execution_manager.set_final_context(paused_result, cast(Optional[ContextT], ctx))
        # Emit a post_run event to allow tracers (e.g., ConsoleTracer) to render the final
        # completion panel after a resume completes. The initial run already emitted a
        # post_run reflecting the paused state; this second event represents the true end.
        try:
            await self._dispatch_hook(
                "post_run",
                pipeline_result=paused_result,
                context=ctx,
            )
        except Exception as e:  # noqa: BLE001
            # Never let tracing/hooks break control flow; record for diagnostics
            try:
                from ..infra import telemetry as _telemetry

                _telemetry.logfire.debug(f"post_run hook after resume failed: {e}")
            except Exception:
                pass
        return paused_result

    async def replay_from_trace(self, run_id: str) -> PipelineResult[ContextT]:
        """Replay a prior run deterministically using recorded trace and responses (FSD-013)."""
        # 1) Load trace and step history
        if self.state_backend is None:
            raise OrchestratorError("Replay requires a state_backend with trace support")
        stored = await self.state_backend.get_run_details(run_id)
        steps = await self.state_backend.list_run_steps(run_id)
        trace = await self.state_backend.get_trace(run_id)

        if stored is None:
            raise OrchestratorError(f"No stored run metadata for run_id={run_id}")
        if steps is None:
            steps = []

        # 2) Prepare initial input/context
        initial_input: Any = None
        initial_context_data: Dict[str, Any] = {}
        try:
            # Prefer trace root attributes for input when available
            if isinstance(trace, dict):
                attrs = trace.get("attributes", {}) if trace else {}
                initial_input = attrs.get("flujo.input", None)
        except Exception:
            initial_input = None
        # Fallback to stored state snapshot
        try:
            loaded_state = await self.state_backend.load_state(run_id)
            if loaded_state is not None:
                initial_context_data = loaded_state.get("pipeline_context") or {}
        except Exception:
            initial_context_data = {}

        # 3) Build map of recorded raw responses keyed by step name and attempt
        response_map: Dict[str, Any] = {}
        for s in steps:
            step_name = s.get("step_name", "")
            key = f"{step_name}:attempt_1"
            raw_resp = s.get("raw_response")
            if raw_resp is None:
                # As a fallback, use the output (best-effort) if raw not present
                raw_resp = s.get("output")
            response_map[key] = raw_resp

        # 4) Extract ordered human inputs from trace events
        human_inputs: list[Any] = []

        def _collect_events(span: Dict[str, Any]) -> None:
            try:
                for ev in span.get("events", []) or []:
                    if ev.get("name") == "flujo.resumed":
                        human_inputs.append(ev.get("attributes", {}).get("human_input"))
                for ch in span.get("children", []) or []:
                    _collect_events(ch)
            except Exception:
                pass

        if isinstance(trace, dict):
            _collect_events(trace)

        # 5) Create ReplayAgent
        from ..testing.replay import ReplayAgent

        replay_agent = ReplayAgent(response_map)

        # 6) Override all step agents with the ReplayAgent in-memory
        self._ensure_pipeline()
        assert self.pipeline is not None
        for st in self.pipeline.steps:
            try:
                setattr(st, "agent", replay_agent)
            except Exception:
                pass

        # 7) Patch resume_async to feed human inputs without external IO
        original_resume = self.resume_async

        async def _resume_patched(
            paused_result: PipelineResult[ContextT], human_input: Any
        ) -> PipelineResult[ContextT]:
            # Pull next recorded input
            if not human_inputs:
                raise OrchestratorError("ReplayError: no recorded human input available for resume")
            next_input = human_inputs.pop(0)
            return await original_resume(paused_result, next_input)

        self.resume_async = _resume_patched

        # 8) Execute run with restored input/context
        final_result: PipelineResult[ContextT] | None = None
        async for item in self.run_async(initial_input, initial_context_data=initial_context_data):
            final_result = item
        assert final_result is not None

        # If the pipeline paused, automatically resume using recorded inputs until completion
        from ..domain.models import PipelineContext as _PipelineContext

        while True:
            try:
                ctx = getattr(final_result, "final_pipeline_context", None)
                is_paused = False
                if _PipelineContext is not None and isinstance(ctx, _PipelineContext):
                    is_paused = ctx.scratchpad.get("status") == "paused"
                if not is_paused:
                    break
                # Resume with the patched method (ignores provided human_input and pops from queue)
                final_result = await self.resume_async(final_result, cast(Any, None))
            except Exception:
                break
        return final_result

    def as_step(
        self, name: str, *, inherit_context: bool = True, **kwargs: Any
    ) -> Step[RunnerInT, PipelineResult[ContextT]]:
        """Return this ``Flujo`` runner as a composable :class:`Step`.

        Parameters
        ----------
        name:
            Name of the resulting step.
        **kwargs:
            Additional ``Step`` configuration passed to :class:`Step`.

        Returns
        -------
        Step
            Step that executes this runner when invoked inside another pipeline.
        """

        async def _runner(
            initial_input: Any,
            *,
            context: BaseModel | None = None,
            resources: AppResources | None = None,
        ) -> PipelineResult[ContextT]:
            # CRITICAL FIX: Create deep copies to prevent shared state between concurrent runs
            import copy

            initial_sub_context_data: Dict[str, Any] = {}
            if inherit_context and context is not None:
                initial_sub_context_data = copy.deepcopy(context.model_dump())
                initial_sub_context_data.pop("run_id", None)
                initial_sub_context_data.pop("pipeline_name", None)
                initial_sub_context_data.pop("pipeline_version", None)
            else:
                initial_sub_context_data = copy.deepcopy(self.initial_context_data)

            if "initial_prompt" not in initial_sub_context_data:
                initial_sub_context_data["initial_prompt"] = str(initial_input)

            try:
                self._ensure_pipeline()

                # Validate context model compatibility before creating sub-runner
                if self.context_model is not None and inherit_context and context is not None:
                    # Check if the context model can be initialized with the provided data
                    try:
                        # Create a test instance to validate compatibility
                        test_data = copy.deepcopy(initial_sub_context_data)
                        test_data.pop("run_id", None)
                        test_data.pop("pipeline_name", None)
                        test_data.pop("pipeline_version", None)

                        # Try to create an instance of the context model
                        self.context_model(**test_data)
                    except ValidationError as e:
                        # Extract missing fields from the validation error
                        missing_fields = _extract_missing_fields(e)
                        context_inheritance_error = ContextInheritanceError(
                            missing_fields=missing_fields,
                            parent_context_keys=(
                                list(context.model_dump().keys()) if context else []
                            ),
                            child_model_name=(
                                self.context_model.__name__ if self.context_model else "Unknown"
                            ),
                        )
                        raise context_inheritance_error

                sub_runner = Flujo(
                    self.pipeline,
                    context_model=self.context_model,
                    initial_context_data=initial_sub_context_data,
                    resources=resources or self.resources,
                    usage_limits=self.usage_limits,
                    hooks=self.hooks,
                    backend=self.backend,
                    state_backend=self.state_backend,
                    delete_on_completion=self.delete_on_completion,
                    registry=self.registry,
                    pipeline_name=self.pipeline_name,
                    pipeline_version=self.pipeline_version,
                )

                async for result in sub_runner.run_async(
                    initial_input,
                    initial_context_data=initial_sub_context_data,
                ):
                    pass  # Consume all results to get the last one
                # Best-effort: merge the sub-runner's final context back into the parent context
                try:
                    if (
                        inherit_context
                        and context is not None
                        and hasattr(result, "final_pipeline_context")
                    ):
                        sub_ctx = getattr(result, "final_pipeline_context", None)
                        if sub_ctx is not None:
                            cm = type(context)
                            for fname in getattr(cm, "model_fields", {}):
                                if not hasattr(sub_ctx, fname):
                                    continue
                                new_val = getattr(sub_ctx, fname)
                                if new_val is None:
                                    continue
                                cur_val = getattr(context, fname, None)
                                if isinstance(cur_val, dict) and isinstance(new_val, dict):
                                    try:
                                        cur_val.update(new_val)
                                    except Exception:
                                        setattr(context, fname, new_val)
                                elif isinstance(cur_val, list) and isinstance(new_val, list):
                                    setattr(context, fname, new_val)
                                else:
                                    setattr(context, fname, new_val)
                except Exception:
                    pass
                return result
            except PipelineContextInitializationError as e:
                cause = getattr(e, "__cause__", None)
                # Provide clearer diagnostics on type mismatches as well as missing fields
                if isinstance(cause, ValidationError):
                    try:
                        type_errors = [
                            err for err in cause.errors() if err.get("type") == "type_error"
                        ]
                        if type_errors:
                            field = type_errors[0].get("loc", ("unknown",))[0]
                            expected = type_errors[0].get("ctx", {}).get("expected_type", "unknown")
                            from ..exceptions import ConfigurationError as _CfgErr

                            raise _CfgErr(
                                f"Context inheritance failed: type mismatch for field '{field}'. "
                                f"Expected a type compatible with '{expected}'."
                            ) from e
                    except Exception:
                        pass
                missing_fields = _extract_missing_fields(cause)
                context_inheritance_error = ContextInheritanceError(
                    missing_fields=missing_fields,
                    parent_context_keys=(list(context.model_dump().keys()) if context else []),
                    child_model_name=(
                        self.context_model.__name__ if self.context_model else "Unknown"
                    ),
                )
                raise context_inheritance_error

        # Allow engine-side merge as well for robustness in typed contexts
        return Step.from_callable(_runner, name=name, updates_context=inherit_context, **kwargs)

    async def _shutdown_state_backend(self) -> None:
        """Shutdown the default state backend to avoid lingering worker threads."""
        if not self._owns_state_backend:
            return
        backend = self.state_backend
        if backend is None:
            return
        shutdown = getattr(backend, "shutdown", None)
        if shutdown is None or not callable(shutdown):
            return
        try:
            result = shutdown()
            if inspect.isawaitable(result):
                await result
        except Exception:
            pass


__all__ = [
    "Flujo",
    "InfiniteRedirectError",
    "InfiniteFallbackError",
    "_accepts_param",
    "_extract_missing_fields",
]
