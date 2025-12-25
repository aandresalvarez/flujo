from __future__ import annotations

import copy
import inspect
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TYPE_CHECKING, Optional, TypeVar, Generic

from pydantic import TypeAdapter, ValidationError

from ..utils.async_bridge import run_sync as _run_sync
from ..exceptions import (
    ContextInheritanceError,
    PipelineAbortSignal,
    PipelineContextInitializationError,
)
from ..domain.commands import AgentCommand
from ..domain.dsl.step import Step, StepConfig
from ..domain.models import (
    BaseModel,
    Chunk,
    Failure,
    Paused,
    PipelineContext,
    PipelineResult,
    StepOutcome,
    StepResult,
    Success,
)
from ..domain.processors import AgentProcessors
from ..domain.resources import AppResources
from ..type_definitions.common import JSONObject
from .core.context.context_manager import _extract_missing_fields
from .core.async_iter import aclose_if_possible
from .runner_execution import resume_async_inner
from .run_session import RunSession

if TYPE_CHECKING:  # pragma: no cover
    from .runner import Flujo
    from ..domain.backends import ExecutionBackend

_agent_command_adapter: TypeAdapter[AgentCommand] = TypeAdapter(AgentCommand)

RunnerInT = TypeVar("RunnerInT")
RunnerOutT = TypeVar("RunnerOutT")
ContextT = TypeVar("ContextT", bound=PipelineContext)


class _RunAsyncHandle(Generic[ContextT]):
    """Async iterable that is also awaitable (returns final PipelineResult)."""

    def __init__(self, factory: Callable[[], AsyncIterator[object]]) -> None:
        self._factory = factory

    def __aiter__(self) -> AsyncIterator[object]:
        return _AutoClosingAsyncIterator(self._factory())

    def __await__(self) -> Iterator[object]:
        async def _consume() -> PipelineResult[ContextT]:
            agen = self._factory()
            last_pr: PipelineResult[ContextT] | None = None
            try:
                async for item in agen:
                    if isinstance(item, PipelineResult):
                        last_pr = item
                    elif isinstance(item, StepResult):
                        pr = PipelineResult[ContextT](step_history=[item])
                        last_pr = pr
                if last_pr is None:
                    return PipelineResult()
                return last_pr
            finally:
                try:
                    aclose = getattr(agen, "aclose", None)
                    if callable(aclose):
                        res = aclose()
                        if inspect.isawaitable(res):
                            await res
                except Exception:
                    pass

        return _consume().__await__()


class _AutoClosingAsyncIterator(AsyncIterator[object]):
    """Wrap an async iterator and eagerly close it after yielding a terminal PipelineResult.

    This avoids teardown-time warnings when callers only consume the first PipelineResult
    (common in pause/resume tests and CLI usage).
    """

    def __init__(self, agen: AsyncIterator[object]) -> None:
        self._agen = agen
        self._closed = False

    def __aiter__(self) -> "_AutoClosingAsyncIterator":
        return self

    async def __anext__(self) -> object:
        if self._closed:
            raise StopAsyncIteration
        item = await self._agen.__anext__()
        if isinstance(item, PipelineResult):
            self._closed = True
            await aclose_if_possible(self._agen)
        return item

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await aclose_if_possible(self._agen)


async def run_outcomes_async(
    self: Flujo[RunnerInT, RunnerOutT, ContextT],
    initial_input: RunnerInT,
    *,
    run_id: str | None = None,
    initial_context_data: Optional[JSONObject] = None,
) -> AsyncIterator[StepOutcome[StepResult]]:
    """Run the pipeline and yield typed StepOutcome events."""
    pipeline_result_obj: PipelineResult[ContextT] = PipelineResult()
    last_step_result: StepResult | None = None
    agen = self.run_async(
        initial_input, run_id=run_id, initial_context_data=initial_context_data
    ).__aiter__()
    try:
        async for item in agen:
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
                yield Chunk(data=item)
    except PipelineAbortSignal:
        try:
            ctx = pipeline_result_obj.final_pipeline_context
            msg = None
            if isinstance(ctx, PipelineContext):
                msg = getattr(ctx, "pause_message", None)
        except Exception:
            msg = None
        yield Paused(message=msg or "Paused for HITL")
        return
    finally:
        await aclose_if_possible(agen)

    try:
        if isinstance(pipeline_result_obj, PipelineResult):
            ctx = pipeline_result_obj.final_pipeline_context
            if isinstance(ctx, PipelineContext):
                status = getattr(ctx, "status", None)
                if status == "paused":
                    msg = getattr(ctx, "pause_message", None)
                    yield Paused(message=msg or "Paused for HITL")
                    return
    except Exception:
        pass

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
        try:
            ctx = pipeline_result_obj.final_pipeline_context
            if isinstance(ctx, PipelineContext):
                status = getattr(ctx, "status", None)
                if status == "paused":
                    msg = getattr(ctx, "pause_message", None)
                    yield Paused(message=msg or "Paused for HITL")
                    return
        except Exception:
            pass
        yield Success(step_result=StepResult(name="<no-steps>", success=True))


async def _consume_run_async_to_result(
    self: Flujo[RunnerInT, RunnerOutT, ContextT],
    initial_input: RunnerInT,
    *,
    run_id: str | None = None,
    initial_context_data: Optional[JSONObject] = None,
) -> PipelineResult[ContextT]:
    """Consume run_async and return the final PipelineResult."""
    result: PipelineResult[ContextT] | None = None
    async for item in self.run_async(
        initial_input,
        run_id=run_id,
        initial_context_data=initial_context_data,
    ):
        if isinstance(item, PipelineResult):
            result = item
    if result is None:
        return PipelineResult()

    if self._tracing_manager is not None and getattr(self._tracing_manager, "root_span", None):
        try:
            result.trace_tree = self._tracing_manager.root_span
        except Exception:
            pass

    try:
        if (
            getattr(result, "step_history", None)
            and getattr(result.step_history[-1], "success", False)
            and getattr(result.step_history[-1], "branch_context", None) is not None
        ):
            result.final_pipeline_context = result.step_history[-1].branch_context
    except Exception:
        pass

    try:
        res = getattr(self, "resources", None)
        if res is not None:
            res_cm: object | None = None
            if hasattr(res, "__aexit__"):
                res_cm = res.__aexit__(None, None, None)
            elif hasattr(res, "__exit__"):
                res_cm = res.__exit__(None, None, None)
            if inspect.isawaitable(res_cm):
                await res_cm
    except Exception:
        pass

    try:
        backend = getattr(self, "backend", None)
        executor = getattr(backend, "_executor", None) if backend is not None else None
        shadow_eval = getattr(executor, "_shadow_evaluator", None) if executor is not None else None
        if shadow_eval is not None:
            maybe = getattr(shadow_eval, "maybe_schedule_run", None)
            if callable(maybe):
                maybe(core=executor, result=result, run_id=run_id)
    except Exception:
        pass

    try:
        from ..utils.config import get_settings

        if not bool(getattr(get_settings(), "test_mode", False)) and getattr(
            result, "success", False
        ):
            from ..infra.lockfile import write_lockfile

            pipeline_obj = self._ensure_pipeline()
            write_lockfile(
                path="flujo.lock",
                pipeline=pipeline_obj,
                result=result,
                pipeline_name=self.pipeline_name,
                pipeline_version=self.pipeline_version,
                pipeline_id=self.pipeline_id,
                run_id=run_id,
            )
    except Exception:
        pass

    return result


async def stream_async(
    self: Flujo[RunnerInT, RunnerOutT, ContextT],
    initial_input: RunnerInT,
    *,
    initial_context_data: Optional[JSONObject] = None,
) -> AsyncIterator[object]:
    pipeline = self._ensure_pipeline()
    last_step = pipeline.steps[-1]
    has_stream = hasattr(last_step.agent, "stream")
    if not has_stream:
        final_result: PipelineResult[ContextT] | None = None
        async for item in self.run_async(initial_input, initial_context_data=initial_context_data):
            if isinstance(item, PipelineResult):
                final_result = item
        if final_result is not None:
            yield final_result
    else:
        seen_chunks: set[str] = set()
        async for item in self.run_async(initial_input, initial_context_data=initial_context_data):
            from ..domain.models import Chunk as _Chunk
            from ..domain.models import StepOutcome as _StepOutcome

            if isinstance(item, _Chunk):
                try:
                    k = str(item.data)
                    if k in seen_chunks:
                        continue
                    seen_chunks.add(k)
                except Exception:
                    pass
                yield item.data
            elif isinstance(item, _StepOutcome):
                continue
            else:
                yield item


def run_sync(
    self: Flujo[RunnerInT, RunnerOutT, ContextT],
    initial_input: RunnerInT,
    *,
    run_id: str | None = None,
    initial_context_data: Optional[JSONObject] = None,
) -> PipelineResult[ContextT]:
    return _run_sync(
        _consume_run_async_to_result(
            self,
            initial_input,
            run_id=run_id,
            initial_context_data=initial_context_data,
        ),
        running_loop_error=(
            "Flujo.run() cannot be called from a running event loop. "
            "If you are in an async environment (like Jupyter, FastAPI, or an "
            "`async def` function), you must use the `run_async()` method."
        ),
    )


def as_step(
    self: Flujo[RunnerInT, RunnerOutT, ContextT],
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
    async def _runner(
        initial_input: RunnerInT,
        *,
        context: BaseModel | None = None,
        resources: AppResources | None = None,
    ) -> PipelineResult[ContextT]:
        initial_sub_context_data: JSONObject = {}
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
            runner_cls = type(self)
            self._ensure_pipeline()

            if self.context_model is not None and inherit_context and context is not None:
                try:
                    test_data = copy.deepcopy(initial_sub_context_data)
                    test_data.pop("run_id", None)
                    test_data.pop("pipeline_name", None)
                    test_data.pop("pipeline_version", None)
                    self.context_model(**test_data)
                except ValidationError as e:
                    missing_fields = _extract_missing_fields(e)
                    context_inheritance_error = ContextInheritanceError(
                        missing_fields=missing_fields,
                        parent_context_keys=(list(context.model_dump().keys()) if context else []),
                        child_model_name=(
                            self.context_model.__name__ if self.context_model else "Unknown"
                        ),
                    )
                    raise context_inheritance_error

            sub_runner = runner_cls(
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

            final_result: PipelineResult[ContextT] | None = None
            async for item in sub_runner.run_async(
                initial_input,
                initial_context_data=initial_sub_context_data,
            ):
                pass
                if isinstance(item, PipelineResult):
                    final_result = item
            result = final_result or PipelineResult()
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
            if isinstance(cause, ValidationError):
                try:
                    type_errors = [err for err in cause.errors() if err.get("type") == "type_error"]
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
                child_model_name=(self.context_model.__name__ if self.context_model else "Unknown"),
            )
            raise context_inheritance_error

    return Step.from_callable(
        _runner,
        name=name,
        updates_context=inherit_context,
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


def create_default_backend(self: Flujo[RunnerInT, RunnerOutT, ContextT]) -> "ExecutionBackend":
    """Create a default LocalBackend with properly wired ExecutorCore."""
    executor = self._executor_factory.create_executor()
    return self._backend_factory.create_execution_backend(executor=executor)


def close_runner(self: Flujo[RunnerInT, RunnerOutT, ContextT]) -> None:
    """Synchronously release runner-owned resources.

    In async contexts (when an event loop is running in the current thread), callers must
    use `await runner.aclose()` or `async with Flujo(...)` instead of `runner.close()`.
    """
    _run_sync(
        self.aclose(),
        running_loop_error=(
            "Flujo.close() cannot be called from a running event loop. "
            "Use `await runner.aclose()` or `async with Flujo(...)` instead."
        ),
    )


def make_session(
    self: Flujo[RunnerInT, RunnerOutT, ContextT],
) -> RunSession[RunnerInT, RunnerOutT, ContextT]:
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


async def resume_async(
    self: Flujo[RunnerInT, RunnerOutT, ContextT],
    paused_result: PipelineResult[ContextT],
    human_input: object,
) -> PipelineResult[ContextT]:
    try:
        return await resume_async_inner(self, paused_result, human_input, _agent_command_adapter)
    finally:
        await self._shutdown_state_backend()
