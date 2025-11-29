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
    get_type_hints,
    Literal,
)

# No direct Awaitable usage needed; avoid unused import

from pydantic import ValidationError

from ..exceptions import (
    PipelineContextInitializationError,
    PipelineAbortSignal,
    ContextInheritanceError,
    InfiniteFallbackError,
    InfiniteRedirectError as _InfiniteRedirectError,
)
from ..domain.dsl.step import Step
from ..domain.dsl.pipeline import Pipeline
from ..domain.models import (
    BaseModel,
    PipelineResult,
    StepResult,
    UsageLimits,
    PipelineContext,
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
from ..state import StateBackend
from ..infra.registry import PipelineRegistry

from .core.context_manager import (
    _accepts_param,
    _extract_missing_fields,
)


from .core.hook_dispatcher import _dispatch_hook as _dispatch_hook_impl
from .core.factories import BackendFactory, ExecutorFactory
from .run_plan_resolver import RunPlanResolver
from .run_session import RunSession
from .tracer_resolver import setup_tracing
from .runner_execution import resume_async_inner, replay_from_trace

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
        return await resume_async_inner(self, paused_result, human_input, _agent_command_adapter)

    async def replay_from_trace(self, run_id: str) -> PipelineResult[ContextT]:
        """Replay a prior run deterministically using recorded trace and responses (FSD-013)."""
        return await replay_from_trace(self, run_id)

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
