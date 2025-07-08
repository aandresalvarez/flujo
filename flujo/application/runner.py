from __future__ import annotations

import asyncio
import inspect
import weakref
import copy
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

from pydantic import ValidationError

from ..infra import telemetry
from ..exceptions import (
    OrchestratorError,
    PipelineContextInitializationError,
    UsageLimitExceededError,
    PipelineAbortSignal,
    PausedException,
    TypeMismatchError,
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
)
from ..domain.commands import AgentCommand, ExecutedCommandLog
from pydantic import TypeAdapter
from ..domain.resources import AppResources
from ..domain.types import HookCallable
from ..domain.backends import ExecutionBackend, StepExecutionRequest
from ..tracing import ConsoleTracer
from ..state import StateBackend, WorkflowState
from ..registry import PipelineRegistry

from .context_manager import (
    _accepts_param,
    _extract_missing_fields,
    _types_compatible,
)
from .core.step_logic import _run_step_logic
from .core.context_adapter import _build_context_update, _inject_context
from .core.hook_dispatcher import _dispatch_hook as _dispatch_hook_impl

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
    ) -> None:
        if isinstance(pipeline, Step):
            pipeline = Pipeline.from_step(pipeline)
        self.pipeline: Pipeline[RunnerInT, RunnerOutT] | None = pipeline
        self.registry = registry
        self.pipeline_name = pipeline_name
        self.pipeline_version = pipeline_version
        self.context_model = context_model
        self.initial_context_data: Dict[str, Any] = initial_context_data or {}
        self.resources = resources
        self.usage_limits = usage_limits
        self.hooks = hooks or []
        tracer_instance = None
        if isinstance(local_tracer, ConsoleTracer):
            tracer_instance = local_tracer
        elif local_tracer == "default":
            tracer_instance = ConsoleTracer()
        if tracer_instance:
            self.hooks.append(tracer_instance.hook)
        if backend is None:
            from ..infra.backends import LocalBackend

            backend = LocalBackend()
        self.backend = backend
        self.state_backend = state_backend
        self.delete_on_completion = delete_on_completion

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

    async def _run_step(
        self,
        step: Step[Any, Any],
        data: Any,
        context: Optional[ContextT],
        resources: Optional[AppResources],
        *,
        stream: bool = False,
    ) -> AsyncIterator[Any]:
        """Execute a single step and update context if required.

        Parameters
        ----------
        step:
            The :class:`Step` to execute.
        data:
            Input data for the step.
        context:
            Current pipeline context instance or ``None``.
        resources:
            Application resources passed to the step.

        Returns
        -------
        StepResult
            Result object describing the step outcome.

        Notes
        -----
        If ``step`` is configured with ``updates_context=True`` the returned
        output is merged into ``context`` and revalidated against the context
        model. Validation errors are logged and cause the step to be marked as
        failed.
        """
        q: asyncio.Queue[Any] | None = None

        async def _capture(chunk: Any) -> None:
            assert q is not None
            await q.put(chunk)

        request = StepExecutionRequest(
            step=step,
            input_data=data,
            context=context,
            resources=resources,
            context_model_defined=self.context_model is not None,
            usage_limits=self.usage_limits,
            stream=stream,
            on_chunk=_capture if stream else None,
        )

        if stream:
            q = asyncio.Queue()
            task = asyncio.create_task(self.backend.execute_step(request))
            while True:
                if not q.empty():
                    yield q.get_nowait()
                    continue
                if task.done():
                    while not q.empty():
                        yield q.get_nowait()
                    try:
                        result = task.result()
                    except Exception as e:  # pragma: no cover - defensive
                        telemetry.logfire.error(
                            f"Streaming task for step '{step.name}' failed: {e}"
                        )
                        result = StepResult(
                            name=step.name,
                            output=None,
                            success=False,
                            attempts=1,
                            feedback=str(e),
                        )
                    break
                try:
                    item = await asyncio.wait_for(q.get(), timeout=0.1)
                    yield item
                except asyncio.TimeoutError:
                    continue
        else:
            result = await self.backend.execute_step(request)
        if getattr(step, "updates_context", False):
            if self.context_model is not None and context is not None:
                update_data = _build_context_update(result.output)
                if update_data is None:
                    telemetry.logfire.warn(
                        f"Step '{step.name}' has updates_context=True but did not return a dict or Pydantic model. "
                        "Skipping context update."
                    )
                    yield result
                    return

                err = _inject_context(context, update_data, self.context_model)
                if err is not None:
                    error_msg = (
                        f"Context update by step '{step.name}' failed Pydantic validation: {err}"
                    )
                    telemetry.logfire.error(error_msg)
                    result.success = False
                    result.feedback = error_msg
                    yield result
                    return

                telemetry.logfire.info(
                    f"Context successfully updated and re-validated by step '{step.name}'."
                )
        yield result

    def _check_usage_limits(
        self,
        pipeline_result: PipelineResult[ContextT],
        span: Any | None,
    ) -> None:
        """Enforce token and cost limits for the current run.

        Parameters
        ----------
        pipeline_result:
            The aggregated :class:`PipelineResult` so far.
        span:
            Optional telemetry span used to annotate governor breaches.

        Raises
        ------
        UsageLimitExceededError
            If either cost or token limits configured in ``usage_limits`` are
            exceeded.
        """
        if self.usage_limits is None:
            return

        total_tokens = sum(sr.token_counts for sr in pipeline_result.step_history)

        if (
            self.usage_limits.total_cost_usd_limit is not None
            and pipeline_result.total_cost_usd > self.usage_limits.total_cost_usd_limit
        ):
            if span is not None:
                try:
                    span.set_attribute("governor_breached", True)
                except Exception as e:
                    # Defensive: log and ignore errors setting span attributes
                    telemetry.logfire.error(f"Error setting span attribute: {e}")
                telemetry.logfire.warn(
                    f"Cost limit of ${self.usage_limits.total_cost_usd_limit} exceeded"
                )
                raise UsageLimitExceededError(
                    f"Cost limit of ${self.usage_limits.total_cost_usd_limit} exceeded",
                    pipeline_result,
                )

        if (
            self.usage_limits.total_tokens_limit is not None
            and total_tokens > self.usage_limits.total_tokens_limit
        ):
            if span is not None:
                try:
                    span.set_attribute("governor_breached", True)
                except Exception as e:
                    # Defensive: log and ignore errors setting span attributes
                    telemetry.logfire.error(f"Error setting span attribute: {e}")
                telemetry.logfire.warn(
                    f"Token limit of {self.usage_limits.total_tokens_limit} exceeded"
                )
                raise UsageLimitExceededError(
                    f"Token limit of {self.usage_limits.total_tokens_limit} exceeded",
                    pipeline_result,
                )

    @staticmethod
    def _set_final_context(result: PipelineResult[ContextT], ctx: Optional[ContextT]) -> None:
        """Write ``ctx`` into ``result`` if present."""

        if ctx is not None:
            result.final_pipeline_context = ctx

    async def _persist_workflow_state(
        self,
        *,
        run_id: str | None,
        context: Optional[ContextT],
        current_step_index: int,
        last_step_output: Any | None,
        status: Literal["running", "paused", "completed", "failed", "cancelled"],
        state_created_at: datetime | None,
        state_backend: StateBackend | None = None,
    ) -> None:
        """Persist workflow state if a backend is configured."""

        backend = state_backend or self.state_backend
        if backend is None or not run_id or context is None:
            return

        self._ensure_pipeline()
        wf_state = WorkflowState(
            run_id=run_id,
            pipeline_id=str(id(self.pipeline)),
            pipeline_name=self.pipeline_name or "",
            pipeline_version=self.pipeline_version,
            current_step_index=current_step_index,
            pipeline_context=context.model_dump(),
            last_step_output=last_step_output,
            status=status,
            created_at=state_created_at or datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        await backend.save_state(run_id, wf_state.model_dump())
        if self.delete_on_completion and status in {"completed", "failed"}:
            await backend.delete_state(run_id)

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
        """Iterate over pipeline steps yielding streaming output.

        Parameters
        ----------
        start_idx:
            Index of the first step to execute.
        data:
            Input to the first step.
        context:
            Mutable context object passed between steps.
        result:
            Aggregated :class:`PipelineResult` to populate.
        stream_last:
            If ``True`` and the final step supports ``stream``, yield chunks as
            they are produced.

        Yields
        ------
        Any
            Streaming output chunks from the final step when ``stream_last`` is
            enabled.

        Raises
        ------
        PausedException
            If a step pauses the pipeline for human input.
        TypeMismatchError
            If the output type of a step does not match the next step's
            expected input type.
        """
        assert self.pipeline is not None
        for idx, step in enumerate(self.pipeline.steps[start_idx:], start=start_idx):
            await self._dispatch_hook(
                "pre_step",
                step=step,
                step_input=data,
                context=context,
                resources=self.resources,
            )
            with telemetry.logfire.span(step.name) as span:
                try:
                    is_last = idx == len(self.pipeline.steps) - 1
                    if (
                        stream_last
                        and is_last
                        and step.agent is not None
                        and hasattr(step.agent, "stream")
                    ):
                        async for item in self._run_step(
                            step,
                            data,
                            context=context,
                            resources=self.resources,
                            stream=True,
                        ):
                            if isinstance(item, StepResult):
                                step_result = item
                            else:
                                yield item
                    else:
                        async for item in self._run_step(
                            step,
                            data,
                            context=context,
                            resources=self.resources,
                        ):
                            step_result = cast(StepResult, item)
                except PausedException as e:
                    if isinstance(context, PipelineContext):
                        context.scratchpad["status"] = "paused"
                        context.scratchpad["pause_message"] = str(e)
                        scratch = context.scratchpad
                        if "paused_step_input" not in scratch:
                            scratch["paused_step_input"] = data
                    self._set_final_context(result, context)
                    break
                if step_result.metadata_:
                    for key, value in step_result.metadata_.items():
                        try:
                            span.set_attribute(key, value)
                        except Exception as e:
                            telemetry.logfire.error(f"Error setting span attribute: {e}")
                result.step_history.append(step_result)
                result.total_cost_usd += step_result.cost_usd
                self._check_usage_limits(result, span)
            if step_result.success:
                await self._dispatch_hook(
                    "post_step",
                    step_result=step_result,
                    context=context,
                    resources=self.resources,
                )
                if state_backend is not None and context is not None and run_id is not None:
                    await self._persist_workflow_state(
                        run_id=run_id,
                        context=context,
                        current_step_index=idx + 1,
                        last_step_output=step_result.output,
                        status="running",
                        state_created_at=state_created_at,
                        state_backend=state_backend,
                    )
            else:
                await self._dispatch_hook(
                    "on_step_failure",
                    step_result=step_result,
                    context=context,
                    resources=self.resources,
                )
                telemetry.logfire.warn(f"Step '{step.name}' failed. Halting pipeline execution.")
                break
            if idx < len(self.pipeline.steps) - 1:
                next_step = self.pipeline.steps[idx + 1]
                expected = getattr(next_step, "__step_input_type__", Any)
                actual_type = type(step_result.output)
                if step_result.output is None:
                    actual_type = type(None)
                if not _types_compatible(actual_type, expected):
                    raise TypeMismatchError(
                        f"Type mismatch: Output of '{step.name}' (returns `{actual_type}`) "
                        f"is not compatible with '{next_step.name}' (expects `{expected}`). "
                        "For best results, use a static type checker like mypy to catch these issues before runtime."
                    )
            data = step_result.output

    async def run_async(
        self,
        initial_input: RunnerInT,
        *,
        initial_context_data: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[PipelineResult[ContextT]]:
        """Run the pipeline asynchronously.

        This method should be used when an asyncio event loop is already
        running, such as within Jupyter notebooks or async web frameworks.

        It yields any streaming output from the final step and then the final
        ``PipelineResult`` object.
        """
        current_context_instance: Optional[ContextT] = None
        if self.context_model is not None:
            try:
                context_data = {**self.initial_context_data}
                if initial_context_data:
                    context_data.update(initial_context_data)
                current_context_instance = self.context_model(**context_data)
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
            current_context_instance = cast(
                ContextT,
                PipelineContext(initial_prompt=str(initial_input)),
            )

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
        run_id_for_state = getattr(current_context_instance, "run_id", None)
        if self.state_backend is not None and run_id_for_state:
            loaded = await self.state_backend.load_state(run_id_for_state)
            if loaded is not None:
                wf_state = WorkflowState.model_validate(loaded)
                self.pipeline_name = wf_state.pipeline_name
                self.pipeline_version = wf_state.pipeline_version
                self._ensure_pipeline()
                start_idx = wf_state.current_step_index
                state_created_at = wf_state.created_at
                assert self.pipeline is not None
                if start_idx > len(self.pipeline.steps):
                    raise OrchestratorError(
                        f"Invalid persisted step index {start_idx} for pipeline with {len(self.pipeline.steps)} steps"
                    )
                if wf_state.pipeline_context is not None:
                    if self.context_model is not None:
                        current_context_instance = self.context_model.model_validate(
                            wf_state.pipeline_context
                        )
                    else:
                        current_context_instance = cast(
                            ContextT, PipelineContext.model_validate(wf_state.pipeline_context)
                        )
                if start_idx > 0:
                    data = wf_state.last_step_output
            await self._persist_workflow_state(
                run_id=run_id_for_state,
                context=current_context_instance,
                current_step_index=start_idx,
                last_step_output=data,
                status="running",
                state_created_at=state_created_at,
                state_backend=self.state_backend,
            )
        else:
            self._ensure_pipeline()
        cancelled = False
        try:
            await self._dispatch_hook(
                "pre_run",
                initial_input=initial_input,
                context=current_context_instance,
                resources=self.resources,
            )
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
                yield chunk
        except asyncio.CancelledError:
            telemetry.logfire.info("Pipeline cancelled")
            cancelled = True
            yield pipeline_result_obj
            return
        except PipelineAbortSignal as e:
            telemetry.logfire.info(str(e))
        except UsageLimitExceededError:
            if current_context_instance is not None:
                self._set_final_context(
                    pipeline_result_obj,
                    cast(Optional[ContextT], current_context_instance),
                )
            raise
        finally:
            if current_context_instance is not None:
                self._set_final_context(
                    pipeline_result_obj,
                    cast(Optional[ContextT], current_context_instance),
                )
                final_status: Literal[
                    "running",
                    "paused",
                    "completed",
                    "failed",
                    "cancelled",
                ]
                if cancelled:
                    final_status = "cancelled"
                elif pipeline_result_obj.step_history:
                    final_status = (
                        "completed"
                        if all(s.success for s in pipeline_result_obj.step_history)
                        else "failed"
                    )
                else:
                    final_status = "failed"
                if isinstance(current_context_instance, PipelineContext):
                    if current_context_instance.scratchpad.get("status") == "paused":
                        final_status = "paused"
                    current_context_instance.scratchpad["status"] = final_status
                await self._persist_workflow_state(
                    run_id=getattr(current_context_instance, "run_id", None),
                    context=current_context_instance,
                    current_step_index=start_idx + len(pipeline_result_obj.step_history),
                    last_step_output=(
                        pipeline_result_obj.step_history[-1].output
                        if pipeline_result_obj.step_history
                        else None
                    ),
                    status=final_status,
                    state_created_at=state_created_at,
                )
            try:
                await self._dispatch_hook(
                    "post_run",
                    pipeline_result=pipeline_result_obj,
                    context=current_context_instance,
                    resources=self.resources,
                )
            except PipelineAbortSignal as e:
                telemetry.logfire.info(str(e))

        yield pipeline_result_obj
        return

    async def stream_async(
        self,
        initial_input: RunnerInT,
        *,
        initial_context_data: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Any]:
        async for item in self.run_async(initial_input, initial_context_data=initial_context_data):
            yield item

    def run(
        self,
        initial_input: RunnerInT,
        *,
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
            async for item in self.run_async(
                initial_input, initial_context_data=initial_context_data
            ):
                result = item  # last yield is the PipelineResult
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

        paused_step_result = StepResult(
            name=paused_step.name,
            output=human_input,
            success=True,
            attempts=1,
        )
        if isinstance(ctx, PipelineContext):
            pending = ctx.scratchpad.pop("paused_step_input", None)
            if pending is not None:
                try:
                    pending_cmd = _agent_command_adapter.validate_python(pending)
                except ValidationError:
                    pending_cmd = None
                if pending_cmd is not None:
                    log_entry = ExecutedCommandLog(
                        turn=len(ctx.command_log) + 1,
                        generated_command=pending_cmd,
                        execution_result=human_input,
                    )
                    ctx.command_log.append(log_entry)
        paused_result.step_history.append(paused_step_result)

        data = human_input
        run_id_for_state = getattr(ctx, "run_id", None)
        state_created_at: datetime | None = None
        if self.state_backend is not None and run_id_for_state is not None:
            loaded = await self.state_backend.load_state(run_id_for_state)
            if loaded is not None:
                wf_state_loaded = WorkflowState.model_validate(loaded)
                state_created_at = wf_state_loaded.created_at
        async for _ in self._execute_steps(
            start_idx + 1,
            data,
            cast(Optional[ContextT], ctx),
            paused_result,
            stream_last=False,
            run_id=run_id_for_state,
            state_backend=self.state_backend,
            state_created_at=state_created_at,
        ):
            pass

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

        await self._persist_workflow_state(
            run_id=run_id_for_state,
            context=ctx,
            current_step_index=len(paused_result.step_history),
            last_step_output=(
                paused_result.step_history[-1].output if paused_result.step_history else None
            ),
            status=final_status,
            state_created_at=state_created_at,
        )

        self._set_final_context(paused_result, cast(Optional[ContextT], ctx))
        return paused_result

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
            initial_sub_context_data: Dict[str, Any] = {}
            if inherit_context and context is not None:
                initial_sub_context_data = context.model_dump()
            else:
                initial_sub_context_data = copy.deepcopy(self.initial_context_data)

            if "initial_prompt" not in initial_sub_context_data:
                initial_sub_context_data["initial_prompt"] = str(initial_input)

            try:
                self._ensure_pipeline()
                sub_runner = Flujo(
                    self.pipeline,
                    context_model=self.context_model,
                    initial_context_data=initial_sub_context_data,
                    resources=resources or self.resources,
                    usage_limits=self.usage_limits,
                    hooks=self.hooks,
                    backend=self.backend,
                    registry=self.registry,
                    pipeline_name=self.pipeline_name,
                    pipeline_version=self.pipeline_version,
                )
            except PipelineContextInitializationError as e:
                cause = getattr(e, "__cause__", None)
                missing_fields = _extract_missing_fields(cause)
                raise ContextInheritanceError(
                    missing_fields=missing_fields,
                    parent_context_keys=(list(context.model_dump().keys()) if context else []),
                    child_model_name=(
                        self.context_model.__name__ if self.context_model else "Unknown"
                    ),
                ) from e
            final_result: PipelineResult[ContextT] | None = None
            try:
                async for item in sub_runner.run_async(initial_input):
                    final_result = item
            except PipelineContextInitializationError as e:
                cause = getattr(e, "__cause__", None)
                missing_fields = _extract_missing_fields(cause)
                raise ContextInheritanceError(
                    missing_fields=missing_fields,
                    parent_context_keys=(list(context.model_dump().keys()) if context else []),
                    child_model_name=(
                        self.context_model.__name__ if self.context_model else "Unknown"
                    ),
                ) from e
            if final_result is None:
                raise OrchestratorError(
                    "Final result is None. The pipeline did not produce a valid result."
                )
            if inherit_context and context is not None and final_result.final_pipeline_context:
                context.__dict__.update(final_result.final_pipeline_context.__dict__)
            return final_result

        return Step.from_callable(_runner, name=name, **kwargs)


__all__ = [
    "Flujo",
    "InfiniteRedirectError",
    "InfiniteFallbackError",
    "_run_step_logic",
    "_accepts_param",
    "_extract_missing_fields",
]
