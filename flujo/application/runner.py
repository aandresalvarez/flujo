from __future__ import annotations

import asyncio
import inspect
import weakref
import os
from datetime import datetime
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
    PipelineAbortSignal,
    ContextInheritanceError,
    InfiniteFallbackError,
    InfiniteRedirectError as _InfiniteRedirectError,
)
from ..domain.dsl.step import Step
from ..domain.dsl.pipeline import Pipeline
from ..domain.dsl.step import HumanInTheLoopStep
from ..domain.models import (
    BaseModel,
    PipelineResult,
    StepResult,
    UsageLimits,
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
from ..domain.interfaces import StateProvider
from ..state import StateBackend, WorkflowState
from ..infra.registry import PipelineRegistry

from .core.context_manager import (
    _accepts_param,
    _extract_missing_fields,
)


from .core.hook_dispatcher import _dispatch_hook as _dispatch_hook_impl
from .core.execution_manager import ExecutionManager
from .core.factories import BackendFactory, ExecutorFactory
from .core.state_manager import StateManager
from .run_plan_resolver import RunPlanResolver
from .run_session import RunSession
from .tracer_resolver import setup_tracing

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

_CtxT = TypeVar("_CtxT", bound=PipelineContext)


class _RunAsyncHandle(Generic[_CtxT]):
    """Async iterable that is also awaitable (returns final PipelineResult)."""

    def __init__(self, factory: Callable[[], AsyncIterator[PipelineResult[_CtxT]]]) -> None:
        self._factory = factory

    def __aiter__(self) -> AsyncIterator[PipelineResult[_CtxT]]:
        return self._factory()

    def __await__(self):
        async def _consume() -> PipelineResult[_CtxT]:
            agen = self._factory()
            last_pr: PipelineResult[_CtxT] | None = None
            try:
                async for item in agen:
                    if isinstance(item, PipelineResult):
                        last_pr = item
                    elif isinstance(item, StepResult):
                        # Wrap step result into a PipelineResult-like view
                        pr = PipelineResult[_CtxT](step_history=[item])
                        last_pr = pr
                if last_pr is None:
                    return PipelineResult()
                return last_pr
            finally:
                try:
                    await agen.aclose()
                except Exception:
                    pass

        return _consume().__await__()


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
ContextT = TypeVar("ContextT", bound=PipelineContext)


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
    state_providers : Dict[str, StateProvider], optional
        External state providers for :class:`ContextReference` hydration. Ignored when a
        custom ``executor_factory`` is supplied; pass providers directly to the factory
        instead.
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
        executor_factory: Optional[ExecutorFactory] = None,
        backend_factory: Optional[BackendFactory] = None,
        pipeline_version: str = "latest",
        local_tracer: Union[str, Any, None] = None,
        registry: Optional[PipelineRegistry] = None,
        pipeline_name: Optional[str] = None,
        enable_tracing: bool = True,
        pipeline_id: Optional[str] = None,
        state_providers: Optional[Dict[str, StateProvider]] = None,
    ) -> None:
        if isinstance(pipeline, Step):
            pipeline = Pipeline.from_step(pipeline)
        self.pipeline: Pipeline[RunnerInT, RunnerOutT] | None = pipeline
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
        self.registry = registry
        self._plan_resolver: RunPlanResolver[RunnerInT, RunnerOutT] = RunPlanResolver(
            pipeline=pipeline,
            registry=registry,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
        )
        self.pipeline = self._plan_resolver.pipeline
        self.pipeline_name = self._plan_resolver.pipeline_name
        self.pipeline_version = self._plan_resolver.pipeline_version
        self.context_model = context_model
        self.initial_context_data: Dict[str, Any] = initial_context_data or {}
        self.resources = resources

        def _post_run_only(hook: HookCallable) -> HookCallable:
            async def _wrapped(payload: Any) -> None:
                if getattr(payload, "event_name", None) == "post_run":
                    await hook(payload)

            return _wrapped

        pipeline_hooks: list[HookCallable] = []
        pipeline_finish_hooks: list[HookCallable] = []
        try:
            if self.pipeline is not None:
                pipeline_hooks.extend(list(getattr(self.pipeline, "hooks", []) or []))
                pipeline_finish_hooks.extend(
                    [_post_run_only(h) for h in getattr(self.pipeline, "on_finish", []) or []]
                )
        except Exception:
            pipeline_hooks = []
            pipeline_finish_hooks = []

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

        # Store state providers for ContextReference hydration
        self._state_providers = state_providers or {}

        # Handle executor factory creation with state_providers support
        if executor_factory is not None:
            self._executor_factory = executor_factory
            # Warn if user also provided state_providers (they'll be ignored)
            if self._state_providers:
                warnings.warn(
                    "state_providers is ignored when executor_factory is provided. "
                    "Pass state_providers to your ExecutorFactory instead.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            # Create executor factory with state_providers
            self._executor_factory = ExecutorFactory(state_providers=self._state_providers)

        # Handle backend factory - ensure state_providers propagate even with custom backend_factory
        self._backend_factory = backend_factory or BackendFactory(self._executor_factory)
        # If a custom backend_factory is supplied, align its executor factory so that any
        # internally created executors also receive the configured state_providers.
        if backend_factory is not None and hasattr(self._backend_factory, "_executor_factory"):
            self._backend_factory._executor_factory = self._executor_factory

        combined_hooks: list[HookCallable] = []
        combined_hooks.extend(pipeline_hooks)
        combined_hooks.extend(pipeline_finish_hooks)
        if hooks:
            combined_hooks.extend(list(hooks))
        self.hooks: list[Any] = combined_hooks
        # Centralized tracing/bootstrap: helper attaches trace manager and console tracer hooks.
        self._trace_manager = setup_tracing(
            enable_tracing=enable_tracing,
            local_tracer=local_tracer,
            hooks=self.hooks,
        )
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
        self._owns_state_backend: bool = state_backend is None
        self.delete_on_completion = delete_on_completion
        self._pending_close_tasks: list[asyncio.Task[Any]] = []

    def _create_default_backend(self) -> "ExecutionBackend":
        """Create a default LocalBackend with properly wired ExecutorCore.

        This method acts as the Composition Root, assembling all the
        components needed for optimal execution.

        Note: We explicitly pass an executor created from our _executor_factory
        to ensure state_providers are honored even when a custom backend_factory
        is provided.
        """
        # Create executor from our factory (which has state_providers configured)
        # and pass it to backend_factory to ensure state_providers propagate
        executor = self._executor_factory.create_executor()
        return self._backend_factory.create_execution_backend(executor=executor)

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
        # Wait for background tasks if executor supports it
        try:
            executor = getattr(self.backend, "_executor", None)
            if executor and hasattr(executor, "wait_for_background_tasks"):
                await executor.wait_for_background_tasks()
        except Exception:
            pass

        await self._shutdown_state_backend()

    def close(self) -> None:
        """Synchronously release runner-owned resources (best-effort in async contexts)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose())
        else:
            task = loop.create_task(self.aclose())
            self._pending_close_tasks.append(task)
            task.add_done_callback(self._handle_close_task_done)

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
    ) -> Literal[False]:
        self.close()
        return False

    def _handle_close_task_done(self, task: asyncio.Task[Any]) -> None:
        """Detach finished close tasks and surface unexpected errors."""
        try:
            self._pending_close_tasks.remove(task)
        except ValueError:
            pass
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        except Exception:
            return
        if exc is not None:
            try:
                from ..infra import telemetry as _telemetry

                _telemetry.logfire.warning(f"Runner close task raised: {exc}")
            except Exception:
                pass

    def _ensure_pipeline(self) -> Pipeline[RunnerInT, RunnerOutT]:
        """Load the configured pipeline from the registry if needed."""
        pipeline = self._plan_resolver.ensure_pipeline()
        self.pipeline = pipeline
        self.pipeline_name = self._plan_resolver.pipeline_name
        self.pipeline_version = self._plan_resolver.pipeline_version
        return pipeline

    def _get_pipeline_meta(self) -> tuple[Optional[str], str]:
        """Current pipeline metadata backing the resolver."""
        return self.pipeline_name, self.pipeline_version

    def _set_pipeline_meta(self, name: Optional[str], version: str) -> None:
        """Synchronize pipeline name/version across resolver and runner."""
        self._plan_resolver.pipeline_name = name
        self._plan_resolver.pipeline_version = version
        self.pipeline_name = name
        self.pipeline_version = version

    def _make_session(self) -> RunSession[RunnerInT, RunnerOutT, ContextT]:
        """Factory for a per-run session composed from the resolver and backends."""
        return RunSession(
            pipeline=self.pipeline,
            pipeline_name=self.pipeline_name,
            pipeline_version=self.pipeline_version,
            pipeline_id=self.pipeline_id,
            context_model=self.context_model,
            initial_context_data=self.initial_context_data,
            resources=self.resources,
            usage_limits=self.usage_limits,
            hooks=self.hooks,
            backend=self.backend,
            state_backend=self.state_backend,
            delete_on_completion=self.delete_on_completion,
            trace_manager=self._trace_manager,
            ensure_pipeline=self._ensure_pipeline,
            refresh_pipeline_meta=self._get_pipeline_meta,
            dispatch_hook=self._dispatch_hook,
            shutdown_state_backend=self._shutdown_state_backend,
            set_pipeline_meta=self._set_pipeline_meta,
            reset_pipeline_cache=lambda: setattr(self._plan_resolver, "pipeline", None),
        )

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
        """Delegate step execution to a composed RunSession."""
        session = self._make_session()
        async for item in session.execute_steps(
            start_idx=start_idx,
            data=data,
            context=context,
            result=result,
            stream_last=stream_last,
            run_id=run_id,
            state_backend=state_backend,
            state_created_at=state_created_at,
        ):
            yield item

    async def _run_async_impl(
        self,
        initial_input: RunnerInT,
        *,
        run_id: str | None = None,
        initial_context_data: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[PipelineResult[ContextT]]:
        """Delegate run orchestration to the composed RunSession."""
        session = self._make_session()
        async for item in session.run_async(
            initial_input, run_id=run_id, initial_context_data=initial_context_data
        ):
            if isinstance(item, PipelineResult) and getattr(item, "step_history", None):
                try:
                    last_ctx = getattr(item.step_history[-1], "branch_context", None)
                    if last_ctx is not None:
                        item.final_pipeline_context = last_ctx
                except Exception:
                    pass
            yield item

    def run_async(
        self,
        initial_input: RunnerInT,
        *,
        run_id: str | None = None,
        initial_context_data: Optional[Dict[str, Any]] = None,
    ) -> _RunAsyncHandle[ContextT]:
        """Run the pipeline asynchronously.

        Returns an object that is both async-iterable (streaming) and awaitable
        (returns the final PipelineResult), preserving legacy convenience.
        """

        return _RunAsyncHandle(
            lambda: self._run_async_impl(
                initial_input, run_id=run_id, initial_context_data=initial_context_data
            )
        )

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
                    if (
                        getattr(item, "step_history", None)
                        and getattr(item.step_history[-1], "branch_context", None) is not None
                    ):
                        try:
                            item.final_pipeline_context = item.step_history[-1].branch_context
                        except Exception:
                            pass
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
            try:
                if (
                    getattr(result, "step_history", None)
                    and getattr(result.step_history[-1], "success", False)
                    and getattr(result.step_history[-1], "branch_context", None) is not None
                ):
                    result.final_pipeline_context = result.step_history[-1].branch_context
            except Exception:
                pass
            # Ensure runner-level resources are properly closed
            try:
                if getattr(self, "resources", None):
                    res_cm = None
                    if hasattr(self.resources, "__aexit__"):
                        res_cm = self.resources.__aexit__(None, None, None)
                    elif hasattr(self.resources, "__exit__"):
                        res_cm = self.resources.__exit__(None, None, None)
                    if inspect.isawaitable(res_cm):
                        await res_cm
            except Exception:
                pass
            return result

        return asyncio.run(_consume())

    async def resume_async(
        self, paused_result: PipelineResult[ContextT], human_input: Any
    ) -> PipelineResult[ContextT]:
        """Resume a paused pipeline with human input."""
        try:
            return await self._resume_async_inner(paused_result, human_input)
        finally:
            await self._shutdown_state_backend()

    async def _resume_async_inner(
        self, paused_result: PipelineResult[ContextT], human_input: Any
    ) -> PipelineResult[ContextT]:
        try:
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
                                generated_command=_Ask(
                                    question=scratch.get("pause_message", "Paused")
                                ),
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

        finally:
            # Legacy compatibility: helper should not manage backend lifecycle directly.
            pass

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
        try:
            if (
                final_result is not None
                and getattr(final_result, "step_history", None)
                and getattr(final_result.step_history[-1], "success", False)
                and getattr(final_result.step_history[-1], "branch_context", None) is not None
            ):
                final_result.final_pipeline_context = final_result.step_history[-1].branch_context
        except Exception:
            pass
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
