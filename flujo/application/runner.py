from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import (
    Any,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
    AsyncIterator,
    Union,
    Literal,
)

from ..exceptions import InfiniteFallbackError, InfiniteRedirectError as _InfiniteRedirectError
from ..domain.dsl.step import Step
from ..domain.dsl.pipeline import Pipeline
from ..domain.models import PipelineResult, StepResult, UsageLimits, PipelineContext, StepOutcome
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
from .runner_components import StateBackendManager, TracingManager
from .runner_execution import replay_from_trace
from .runner_methods import (
    _RunAsyncHandle,
    as_step as _as_step,
    create_default_backend as _create_default_backend,
    close_runner as _close_runner,
    make_session as _make_session,
    resume_async as _resume_async,
    run_outcomes_async as _run_outcomes_async,
    run_sync as _run_sync,
    stream_async as _stream_async,
)

import uuid
import warnings
from ..utils.config import get_settings

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

    _tracing_manager: TracingManager
    _state_manager: StateBackendManager
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

        # Tracing lifecycle management
        self._tracing_manager = TracingManager(
            enable_tracing=enable_tracing,
            local_tracer=local_tracer,
        )
        self.hooks = self._tracing_manager.setup(self.hooks)
        self._trace_manager = self._tracing_manager.trace_manager
        if backend is None:
            # ✅ COMPOSITION ROOT: Create and wire all dependencies
            backend = self._create_default_backend()
        self.backend = backend
        # Debug: Log backend and executor type
        from flujo.infra import telemetry

        backend_type = type(self.backend).__name__
        executor_type = getattr(getattr(self.backend, "_executor", None), "__class__", None)
        telemetry.logfire.debug(f"Flujo backend: {backend_type}, executor: {executor_type}")

        self._state_manager = StateBackendManager(
            state_backend=state_backend,
            delete_on_completion=delete_on_completion,
        )
        self.state_backend: StateBackend | None = self._state_manager.backend
        self.delete_on_completion = delete_on_completion
        self._pending_close_tasks: list[asyncio.Task[Any]] = []

    def _create_default_backend(self) -> "ExecutionBackend":
        return _create_default_backend(self)

    def disable_tracing(self) -> None:
        """Disable tracing by removing trace hooks and clearing active manager."""
        self.hooks = self._tracing_manager.disable(self.hooks)
        self._trace_manager = self._tracing_manager.trace_manager

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
        _close_runner(self)

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
        return _make_session(self)

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
        async for outcome in _run_outcomes_async(
            self,
            initial_input,
            run_id=run_id,
            initial_context_data=initial_context_data,
        ):
            yield outcome

    async def stream_async(
        self,
        initial_input: RunnerInT,
        *,
        initial_context_data: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Any]:
        async for item in _stream_async(
            self,
            initial_input,
            initial_context_data=initial_context_data,
        ):
            yield item

    def run(
        self,
        initial_input: RunnerInT,
        *,
        run_id: str | None = None,
        initial_context_data: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult[ContextT]:
        return _run_sync(
            self,
            initial_input,
            run_id=run_id,
            initial_context_data=initial_context_data,
        )

    async def resume_async(
        self, paused_result: PipelineResult[ContextT], human_input: Any
    ) -> PipelineResult[ContextT]:
        return await _resume_async(self, paused_result, human_input)

    async def replay_from_trace(self, run_id: str) -> PipelineResult[ContextT]:
        """Replay a prior run deterministically using recorded trace and responses (FSD-013)."""
        return await replay_from_trace(self, run_id)

    def as_step(
        self, name: str, *, inherit_context: bool = True, **kwargs: Any
    ) -> Step[RunnerInT, PipelineResult[ContextT]]:
        return _as_step(self, name, inherit_context=inherit_context, **kwargs)

    async def _shutdown_state_backend(self) -> None:
        """Shutdown the default state backend to avoid lingering worker threads."""
        await self._state_manager.shutdown()


__all__ = [
    "Flujo",
    "InfiniteRedirectError",
    "InfiniteFallbackError",
    "_accepts_param",
    "_extract_missing_fields",
]
