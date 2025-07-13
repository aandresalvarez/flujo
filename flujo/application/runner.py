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
    ContextInheritanceError,
    InfiniteFallbackError,
    InfiniteRedirectError as _InfiniteRedirectError,
    PausedException,
    MissingAgentError,
    TypeMismatchError,
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
)
from .core.step_logic import _run_step_logic
from .core.context_adapter import _build_context_update, _inject_context
from .core.hook_dispatcher import _dispatch_hook as _dispatch_hook_impl
from .core.execution_manager import ExecutionManager
from .core.state_manager import StateManager
from .core.usage_governor import UsageGovernor
from .core.step_coordinator import StepCoordinator

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
                    except (
                        UsageLimitExceededError,
                        InfiniteFallbackError,
                        InfiniteRedirectError,
                        PausedException,
                        MissingAgentError,
                        TypeMismatchError,
                        TypeError,
                        ValueError,
                        RuntimeError,
                    ):
                        # Allow critical exceptions to propagate
                        raise
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
        usage_governor: UsageGovernor[ContextT] = UsageGovernor[ContextT](self.usage_limits)
        step_coordinator: StepCoordinator[ContextT] = StepCoordinator[ContextT](
            self.hooks, self.resources
        )

        execution_manager = ExecutionManager(
            self.pipeline,
            state_manager=state_manager,
            usage_governor=usage_governor,
            step_coordinator=step_coordinator,
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
            step_executor=self._run_step,
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
        current_context_instance: Optional[ContextT] = None
        if self.context_model is not None:
            try:
                context_data = {**self.initial_context_data}
                if initial_context_data:
                    context_data.update(initial_context_data)
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

                current_context_instance = self.context_model(**processed_context_data)
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
        run_id_for_state = run_id or state_manager.get_run_id_from_context(current_context_instance)

        if run_id_for_state:
            (
                context,
                last_output,
                current_idx,
                created_at,
                pipeline_name,
                pipeline_version,
            ) = await state_manager.load_workflow_state(run_id_for_state, self.context_model)
            if context is not None:
                # Resume from persisted state
                current_context_instance = context
                start_idx = current_idx
                state_created_at = created_at
                if start_idx > 0:
                    data = last_output

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

            # Persist initial state
            await state_manager.persist_workflow_state(
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
                assert self.pipeline is not None
                execution_manager: ExecutionManager[ContextT] = ExecutionManager[ContextT](
                    self.pipeline
                )
                execution_manager.set_final_context(
                    pipeline_result_obj,
                    cast(Optional[ContextT], current_context_instance),
                )
            raise
        finally:
            if current_context_instance is not None:
                assert self.pipeline is not None
                execution_manager = ExecutionManager[ContextT](self.pipeline)
                execution_manager.set_final_context(
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

                # Use execution manager to persist final state
                execution_manager = ExecutionManager[ContextT](
                    self.pipeline,
                    state_manager=state_manager,
                )
                await execution_manager.persist_final_state(
                    run_id=state_manager.get_run_id_from_context(current_context_instance),
                    context=current_context_instance,
                    result=pipeline_result_obj,
                    start_idx=start_idx,
                    state_created_at=state_created_at,
                    final_status=final_status,
                )

                # Delete state if delete_on_completion is True and pipeline completed successfully
                if (
                    self.delete_on_completion
                    and final_status == "completed"
                    and state_manager.get_run_id_from_context(current_context_instance) is not None
                ):
                    await state_manager.delete_workflow_state(
                        state_manager.get_run_id_from_context(current_context_instance)
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
            async for item in self.run_async(
                initial_input,
                run_id=run_id,
                initial_context_data=initial_context_data,
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

        # Delete state if delete_on_completion is True and pipeline completed successfully
        if (
            self.delete_on_completion
            and final_status == "completed"
            and run_id_for_state is not None
        ):
            await state_manager.delete_workflow_state(run_id_for_state)

        execution_manager.set_final_context(paused_result, cast(Optional[ContextT], ctx))
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
                    state_backend=self.state_backend,
                    delete_on_completion=self.delete_on_completion,
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
