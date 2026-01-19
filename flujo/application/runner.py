from __future__ import annotations

import asyncio
import logging
import os
import types
from datetime import datetime
from dataclasses import dataclass
from typing import (
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
    AsyncIterator,
    Literal,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from ..domain.events import HookPayload

from ..exceptions import (
    InfiniteFallbackError,
    InfiniteRedirectError as _InfiniteRedirectError,
    PausedException,
    PipelineAbortSignal,
)
from ..domain.dsl.step import Step, StepConfig
from ..domain.dsl.pipeline import Pipeline
from ..domain.models import (
    Failure,
    Paused,
    PipelineContext,
    PipelineResult,
    StepOutcome,
    StepResult,
    Success,
    UsageLimits,
)
from ..domain.commands import AgentCommand
from pydantic import TypeAdapter
from ..domain.resources import AppResources
from ..domain.processors import AgentProcessors
from ..domain.types import HookCallable
from ..domain.backends import ExecutionBackend
from ..domain.interfaces import StateProvider, RunnerLike
from ..state import StateBackend
from ..infra.registry import PipelineRegistry
from ..type_definitions.common import JSONObject
from .core.context_manager import (
    _accepts_param,
    _extract_missing_fields,
)
from .core.async_iter import aclose_if_possible
from .core.state_manager import StateManager
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
    _consume_run_async_to_result as _consume_run_async_to_result,
    run_sync as _run_sync,
    stream_async as _stream_async,
)

import uuid
import warnings
from ..utils.config import get_settings

logger = logging.getLogger(__name__)

_agent_command_adapter: TypeAdapter[AgentCommand] = TypeAdapter(AgentCommand)


# Alias exported for backwards compatibility
InfiniteRedirectError = _InfiniteRedirectError


RunnerInT = TypeVar("RunnerInT")
RunnerOutT = TypeVar("RunnerOutT")
ContextT = TypeVar("ContextT", bound=PipelineContext)


@dataclass
class BackgroundTaskInfo:
    """Metadata about a persisted background task."""

    task_id: str
    run_id: str
    parent_run_id: Optional[str]
    step_name: Optional[str]
    status: str
    error: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


class _AutoClosingOutcomeIterator(AsyncIterator[StepOutcome[StepResult]]):
    """Auto-close outcome iterators once a terminal outcome is produced.

    Callers often break after the first ``Paused``/``Failure``/``Success`` outcome
    (e.g., pause/resume flows). Async generators do not get implicitly closed on
    early loop exit, which triggers teardown-time RuntimeWarnings under xdist.
    """

    def __init__(self, agen: AsyncIterator[StepOutcome[StepResult]]) -> None:
        self._agen = agen
        self._closed = False

    def __aiter__(self) -> "_AutoClosingOutcomeIterator":
        return self

    async def __anext__(self) -> StepOutcome[StepResult]:
        if self._closed:
            raise StopAsyncIteration
        item = await self._agen.__anext__()
        if isinstance(item, (Success, Failure, Paused)):
            self._closed = True
            await aclose_if_possible(self._agen)
        return item

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await aclose_if_possible(self._agen)


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
    persist_state : bool, default True
        Disable state persistence entirely for ephemeral runs when ``False``.
    state_providers : Dict[str, StateProvider], optional
        External state providers for :class:`ContextReference` hydration. Ignored when a
        custom ``executor_factory`` is supplied; pass providers directly to the factory
        instead.
    """

    _tracing_manager: TracingManager
    _state_manager: StateBackendManager
    _trace_manager: object | None  # Trace manager instance when tracing is enabled
    backend: ExecutionBackend
    state_backend: StateBackend | None
    delete_on_completion: bool
    persist_state: bool

    def __init__(
        self,
        pipeline: (Pipeline[RunnerInT, RunnerOutT] | Step[RunnerInT, RunnerOutT] | None) = None,
        *,
        context_model: Optional[Type[ContextT]] = None,
        initial_context_data: Optional[JSONObject] = None,
        resources: Optional[AppResources] = None,
        usage_limits: Optional[UsageLimits] = None,
        hooks: Optional[list[HookCallable]] = None,
        backend: Optional[ExecutionBackend] = None,
        state_backend: Optional[StateBackend] = None,
        delete_on_completion: bool = False,
        persist_state: bool = True,
        executor_factory: Optional[ExecutorFactory] = None,
        backend_factory: Optional[BackendFactory] = None,
        pipeline_version: str = "latest",
        local_tracer: object | None = None,
        registry: Optional[PipelineRegistry] = None,
        pipeline_name: Optional[str] = None,
        enable_tracing: bool = True,
        pipeline_id: Optional[str] = None,
        state_providers: Optional[Dict[str, StateProvider[object]]] = None,
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
        self.initial_context_data: JSONObject = initial_context_data or {}
        self.resources = resources

        def _post_run_only(hook: HookCallable) -> HookCallable:
            async def _wrapped(payload: HookPayload) -> None:
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
                existing = getattr(self._executor_factory, "_state_providers", None)
                if not existing:
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
        #
        # Avoid mutating a caller-provided BackendFactory instance in-place so the same
        # factory can be safely reused across multiple Flujo runners.
        if backend_factory is not None and hasattr(self._backend_factory, "_executor_factory"):
            try:
                import copy

                self._backend_factory = copy.copy(backend_factory)
            except Exception:
                logger.debug(
                    "Failed to shallow-copy backend_factory; using provided instance.",
                    exc_info=True,
                )
                self._backend_factory = backend_factory
            try:
                self._backend_factory._executor_factory = self._executor_factory
            except Exception:
                logger.debug(
                    "Failed to assign executor_factory on backend_factory.",
                    exc_info=True,
                )
                pass

        combined_hooks: list[HookCallable] = []
        combined_hooks.extend(pipeline_hooks)
        combined_hooks.extend(pipeline_finish_hooks)
        if hooks:
            combined_hooks.extend(list(hooks))
        self.hooks: list[HookCallable] = combined_hooks

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

        self.persist_state = persist_state
        effective_state_backend = state_backend

        # Auto-wire state_backend from FLUJO_STATE_URI if not explicitly provided
        # Skip auto-wiring in test mode to respect test environment expectations
        if effective_state_backend is None and persist_state and not get_settings().test_mode:
            from ..infra.config_manager import get_state_uri

            state_uri = get_state_uri(force_reload=True)
            if state_uri:
                effective_state_backend = self._backend_factory.create_state_backend()

        # Disable state backend if persist_state is False
        effective_state_backend = effective_state_backend if persist_state else None
        if not persist_state and state_backend is not None:
            warnings.warn(
                "persist_state=False ignores the provided state_backend; persistence disabled.",
                UserWarning,
                stacklevel=2,
            )

        self._state_manager = StateBackendManager(
            state_backend=effective_state_backend,
            delete_on_completion=delete_on_completion,
            enable_backend=persist_state,
        )
        self.state_backend: StateBackend | None = self._state_manager.backend
        self.delete_on_completion = delete_on_completion
        self._pending_close_tasks: list[asyncio.Task[object]] = []

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
            if executor:
                if hasattr(executor, "aclose"):
                    await executor.aclose()
                elif hasattr(executor, "wait_for_background_tasks"):
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
        tb: types.TracebackType | None,
    ) -> None:
        await self.aclose()

    def __enter__(self) -> "Flujo[RunnerInT, RunnerOutT, ContextT]":
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
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
        **kwargs: object,
    ) -> None:
        """Invoke registered hooks for ``event_name``."""

        await _dispatch_hook_impl(self.hooks, event_name, **kwargs)

    async def _execute_steps(
        self,
        start_idx: int,
        data: object,
        context: Optional[ContextT],
        result: PipelineResult[ContextT],
        *,
        stream_last: bool = False,
        run_id: str | None = None,
        state_backend: StateBackend | None = None,
        state_created_at: datetime | None = None,
    ) -> AsyncIterator[object]:
        """Delegate step execution to a composed RunSession."""
        session = self._make_session()
        steps_iter = session.execute_steps(
            start_idx=start_idx,
            data=data,
            context=context,
            result=result,
            stream_last=stream_last,
            run_id=run_id,
            state_backend=state_backend,
            state_created_at=state_created_at,
        )
        try:
            async for item in steps_iter:
                yield item
        finally:
            await aclose_if_possible(steps_iter)

    async def _run_async_impl(
        self,
        initial_input: RunnerInT,
        *,
        run_id: str | None = None,
        initial_context_data: Optional[JSONObject] = None,
    ) -> AsyncIterator[object]:
        """Delegate run orchestration to the composed RunSession."""
        session = self._make_session()
        run_iter = session.run_async(
            initial_input, run_id=run_id, initial_context_data=initial_context_data
        )
        try:
            async for item in run_iter:
                if isinstance(item, PipelineResult) and getattr(item, "step_history", None):
                    last_ctx = getattr(item.step_history[-1], "branch_context", None)
                    if last_ctx is not None:
                        item.final_pipeline_context = last_ctx
                yield item
        finally:
            await aclose_if_possible(run_iter)

    def run_async(
        self,
        initial_input: RunnerInT,
        *,
        run_id: str | None = None,
        initial_context_data: Optional[JSONObject] = None,
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

    async def run_result_async(
        self,
        initial_input: RunnerInT,
        *,
        run_id: str | None = None,
        initial_context_data: Optional[JSONObject] = None,
    ) -> PipelineResult[ContextT]:
        """Run the pipeline asynchronously and return the final PipelineResult."""
        return await _consume_run_async_to_result(
            self,
            initial_input,
            run_id=run_id,
            initial_context_data=initial_context_data,
        )

    def run_outcomes_async(
        self,
        initial_input: RunnerInT,
        *,
        run_id: str | None = None,
        initial_context_data: Optional[JSONObject] = None,
    ) -> AsyncIterator[StepOutcome[StepResult]]:
        agen = _run_outcomes_async(
            self,
            initial_input,
            run_id=run_id,
            initial_context_data=initial_context_data,
        )
        return _AutoClosingOutcomeIterator(agen)

    async def run_stream(
        self,
        initial_input: RunnerInT,
        *,
        run_id: str | None = None,
        initial_context_data: Optional[JSONObject] = None,
    ) -> AsyncIterator[StepOutcome[StepResult]]:
        """Run the pipeline and stream StepOutcome events as they complete."""
        async for outcome in self.run_outcomes_async(
            initial_input,
            run_id=run_id,
            initial_context_data=initial_context_data,
        ):
            yield outcome

    async def run_outcomes(
        self,
        initial_input: RunnerInT,
        *,
        run_id: str | None = None,
        initial_context_data: Optional[JSONObject] = None,
    ) -> AsyncIterator[StepOutcome[StepResult]]:
        """Alias for run_stream() with clearer naming."""
        async for outcome in self.run_stream(
            initial_input,
            run_id=run_id,
            initial_context_data=initial_context_data,
        ):
            yield outcome

    async def stream_async(
        self,
        initial_input: RunnerInT,
        *,
        initial_context_data: Optional[JSONObject] = None,
    ) -> AsyncIterator[object]:
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
        initial_context_data: Optional[JSONObject] = None,
    ) -> PipelineResult[ContextT]:
        return _run_sync(
            self,
            initial_input,
            run_id=run_id,
            initial_context_data=initial_context_data,
        )

    async def resume_async(
        self, paused_result: PipelineResult[ContextT], human_input: object
    ) -> PipelineResult[ContextT]:
        return await _resume_async(self, paused_result, human_input)

    async def replay_from_trace(self, run_id: str) -> PipelineResult[ContextT]:
        """Replay a prior run deterministically using recorded trace and responses (FSD-013)."""
        return await replay_from_trace(self, run_id)

    @staticmethod
    def _parse_timestamp(value: object) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except Exception:
                return None
        return None

    def _find_step_by_name(self, name: str) -> Optional[Step[object, object]]:
        if self.pipeline is None:
            return None
        LoopStepType: type[object] | None = None
        try:
            from ..domain.dsl.loop import LoopStep as _LoopStep

            LoopStepType = _LoopStep
        except Exception:
            pass

        queue: list[object] = list(getattr(self.pipeline, "steps", []))
        visited: set[int] = set()

        while queue:
            current_obj = queue.pop(0)
            if not isinstance(current_obj, Step):
                continue
            current: Step[object, object] = current_obj
            if id(current) in visited:
                continue
            visited.add(id(current))

            if getattr(current, "name", None) == name:
                return current

            if (
                LoopStepType is not None
                and isinstance(current, LoopStepType)
                and getattr(current, "body", None)
            ):
                body = getattr(current, "body", None)
                if isinstance(body, Step):
                    queue.append(body)

            branches = getattr(current, "branches", None)
            if branches:
                queue.extend(list(branches))

            nested_steps = getattr(current, "steps", None)
            if nested_steps:
                queue.extend(list(nested_steps))

        return None

    async def get_failed_background_tasks(
        self,
        parent_run_id: Optional[str] = None,
        hours_back: int = 24,
    ) -> list[BackgroundTaskInfo]:
        """List failed background tasks for optional parent run."""
        if self.state_backend is None:
            return []

        tasks = await self.state_backend.get_failed_background_tasks(
            parent_run_id=parent_run_id, hours_back=hours_back
        )
        results: list[BackgroundTaskInfo] = []
        for task in tasks:
            metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
            task_id = metadata.get("task_id") or task.get("task_id")
            if not task_id:
                continue
            results.append(
                BackgroundTaskInfo(
                    task_id=str(task_id),
                    run_id=str(task.get("run_id")),
                    parent_run_id=metadata.get("parent_run_id") or task.get("parent_run_id"),
                    step_name=metadata.get("step_name"),
                    status=str(task.get("status")),
                    error=metadata.get("background_error") or task.get("background_error"),
                    created_at=self._parse_timestamp(task.get("created_at")),
                    updated_at=self._parse_timestamp(task.get("updated_at")),
                )
            )
        return results

    async def resume_background_task(
        self,
        task_id: str,
        new_data: object | None = None,
    ) -> PipelineResult[ContextT]:
        """Resume a failed background task synchronously."""
        if self.state_backend is None:
            raise ValueError("State backend required for background task resumption")

        tasks = await self.state_backend.list_background_tasks(status="failed")
        task = next(
            (
                t
                for t in tasks
                if (t.get("metadata") or {}).get("task_id") == task_id
                or t.get("task_id") == task_id
            ),
            None,
        )
        if task is None:
            raise ValueError(f"Background task '{task_id}' not found")

        bg_run_id = task.get("run_id")
        if bg_run_id is None:
            raise ValueError(f"Background task '{task_id}' missing run_id")

        state_manager = StateManager[ContextT](self.state_backend)
        context, _, _, _, _, _, _ = await state_manager.load_workflow_state(
            bg_run_id, context_model=self.context_model
        )

        metadata_value = task.get("metadata")
        metadata_dict: JSONObject = metadata_value if isinstance(metadata_value, dict) else {}
        data = new_data if new_data is not None else metadata_dict.get("input_data")
        step_name = metadata_dict.get("step_name")
        step = self._find_step_by_name(step_name) if step_name else None
        if step is None:
            raise ValueError(f"Step '{step_name}' not found in pipeline")

        executor = getattr(self.backend, "_executor", None)
        if executor is None:
            raise ValueError("Executor not available for resuming background tasks")

        step_copy = step.model_copy(deep=True)
        if hasattr(step_copy, "config"):
            try:
                step_copy.config.execution_mode = "sync"
            except Exception:
                logger.exception(
                    "Failed to force sync execution_mode when resuming background step",
                    extra={"step_name": step_name},
                )

        final_context = context

        def _context_setter(_res: PipelineResult[ContextT], updated_ctx: ContextT | None) -> None:
            nonlocal final_context
            if updated_ctx is not None:
                final_context = updated_ctx

        frame = executor._make_execution_frame(
            step_copy,
            data,
            context,
            self.resources,
            self.usage_limits,
            _context_setter,
            stream=False,
            on_chunk=None,
            fallback_depth=0,
            quota=None,
            result=None,
        )

        meta_for_persist: JSONObject = dict(metadata_dict)
        meta_for_persist.setdefault("task_id", task_id)
        meta_for_persist.setdefault("is_background_task", True)

        try:
            outcome = await executor.execute(frame)
            step_result = executor._unwrap_outcome_to_step_result(
                outcome, executor._safe_step_name(step_copy)
            )
            if step_result.branch_context is not None:
                final_context = step_result.branch_context
            status = "completed" if getattr(step_result, "success", False) else "failed"
            if status == "completed":
                meta_for_persist.pop("background_error", None)
            await state_manager.persist_workflow_state(
                run_id=bg_run_id,
                context=final_context,
                current_step_index=1,
                last_step_output=getattr(step_result, "output", None),
                status=status,
                step_history=[step_result],
                metadata=meta_for_persist,
            )
            return PipelineResult[ContextT](
                step_history=[step_result],
                final_pipeline_context=final_context,
                success=bool(getattr(step_result, "success", False)),
            )
        except Exception as exc:
            if isinstance(exc, (PausedException, PipelineAbortSignal, InfiniteRedirectError)):
                raise
            meta_for_persist["background_error"] = str(exc)
            await state_manager.persist_workflow_state(
                run_id=bg_run_id,
                context=final_context,
                current_step_index=1,
                last_step_output=None,
                status="failed",
                step_history=[],
                metadata=meta_for_persist,
            )
            raise

    async def retry_failed_background_tasks(
        self,
        parent_run_id: str,
        max_retries: int = 3,
    ) -> list[PipelineResult[ContextT]]:
        """Retry failed background tasks for a given parent run."""
        results: list[PipelineResult[ContextT]] = []
        failures: list[tuple[str, Exception]] = []
        tasks = await self.get_failed_background_tasks(parent_run_id=parent_run_id)
        for task in tasks:
            for attempt in range(max_retries):
                try:
                    result = await self.resume_background_task(task.task_id)
                    results.append(result)
                    break
                except Exception as exc:
                    if isinstance(
                        exc,
                        (PausedException, PipelineAbortSignal, InfiniteRedirectError),
                    ):
                        raise
                    if attempt == max_retries - 1:
                        failures.append((task.task_id, exc))
                        break
                    await asyncio.sleep(2**attempt)
        if failures:
            failure_summary = "; ".join(f"{task_id}: {failure}" for task_id, failure in failures)
            raise RuntimeError(
                f"Retry failed for {len(failures)} background task(s): {failure_summary}"
            )
        return results

    async def cleanup_stale_background_tasks(self, stale_hours: int = 24) -> int:
        """Mark stale running background tasks as failed."""
        if self.state_backend is None:
            return 0
        if hasattr(self.state_backend, "cleanup_stale_background_tasks"):
            try:
                return await self.state_backend.cleanup_stale_background_tasks(stale_hours)
            except Exception:
                logger.exception(
                    "Failed to cleanup stale background tasks",
                    extra={"stale_hours": stale_hours},
                )
                raise
        return 0

    async def run_with_events(
        self,
        initial_input: RunnerInT,
        *,
        run_id: str | None = None,
        initial_context_data: Optional[JSONObject] = None,
    ) -> AsyncIterator[object]:
        """Run pipeline yielding lifecycle events (StepOutcome/Chunk) and final PipelineResult."""
        async for item in self.run_async(
            initial_input, run_id=run_id, initial_context_data=initial_context_data
        ):
            yield item

    def as_step(
        self,
        name: str,
        *,
        inherit_context: bool = True,
        validate_fields: bool = False,
        sink_to: str | None = None,
        processors: AgentProcessors | None = None,
        persist_feedback_to_context: str | None = None,
        persist_validation_results_to: str | None = None,
        is_adapter: bool = False,
        adapter_id: str | None = None,
        adapter_allow: str | None = None,
        config: StepConfig | None = None,
        **config_kwargs: object,
    ) -> Step[RunnerInT, PipelineResult[ContextT]]:
        return _as_step(
            self,
            name,
            inherit_context=inherit_context,
            validate_fields=validate_fields,
            sink_to=sink_to,
            processors=processors,
            persist_feedback_to_context=persist_feedback_to_context,
            persist_validation_results_to=persist_validation_results_to,
            is_adapter=is_adapter,
            adapter_id=adapter_id,
            adapter_allow=adapter_allow,
            config=config,
            **config_kwargs,
        )

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

# Register the default runner factory to break circular dependencies with Pipeline
from ..domain.interfaces import (  # noqa: E402
    set_default_runner_factory as _set_default_runner_factory,
)


def _flujo_factory(
    pipeline: Pipeline[object, object] | Step[object, object] | None = None,
) -> RunnerLike:
    return Flujo(pipeline)


_set_default_runner_factory(_flujo_factory)
