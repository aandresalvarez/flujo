from __future__ import annotations

import asyncio
import copy
import warnings
from datetime import datetime, timezone
from typing import (
    Awaitable,
    AsyncIterator,
    Callable,
    Generic,
    Literal,
    Optional,
    Type,
    TypeVar,
)

from pydantic import ValidationError

from ..domain.backends import ExecutionBackend
from ..domain.dsl.pipeline import Pipeline
from ..domain.models import (
    PipelineContext,
    PipelineResult,
    UsageLimits,
)
from ..domain.resources import AppResources
from ..domain.types import HookCallable
from ..exceptions import (
    ConfigurationError,
    ExecutionError,
    PipelineAbortSignal,
    PipelineContextInitializationError,
    PricingNotConfiguredError,
    UsageLimitExceededError,
)
from ..state import StateBackend
from ..type_definitions.common import JSONObject
from ..utils.config import get_settings
from .core.execution_manager import ExecutionManager
from .core.async_iter import aclose_if_possible
from .core.quota_manager import build_root_quota
from .core.state_manager import StateManager
from .core.step_coordinator import StepCoordinator

ContextT = TypeVar("ContextT", bound=PipelineContext)
RunnerInT = TypeVar("RunnerInT")
RunnerOutT = TypeVar("RunnerOutT")


class RunSession(Generic[RunnerInT, RunnerOutT, ContextT]):
    """Encapsulate a single execution session for a Flujo runner."""

    def _as_context_t(self, ctx: PipelineContext | None) -> Optional[ContextT]:
        if ctx is None:
            return None
        if self.context_model is not None and not isinstance(ctx, self.context_model):
            raise PipelineContextInitializationError(
                f"Expected context of type {self.context_model.__name__}, got {type(ctx).__name__}"
            )
        return ctx  # type: ignore[return-value]

    def __init__(
        self,
        *,
        pipeline: Pipeline[RunnerInT, RunnerOutT] | None,
        pipeline_name: Optional[str],
        pipeline_version: str,
        pipeline_id: str,
        context_model: Optional[Type[ContextT]],
        initial_context_data: JSONObject,
        resources: Optional[AppResources],
        usage_limits: Optional[UsageLimits],
        hooks: list[HookCallable],
        backend: ExecutionBackend,
        state_backend: Optional[StateBackend],
        delete_on_completion: bool,
        trace_manager: object,
        ensure_pipeline: Callable[[], Pipeline[RunnerInT, RunnerOutT]],
        refresh_pipeline_meta: Callable[[], tuple[Optional[str], str]],
        dispatch_hook: Callable[..., Awaitable[None]],
        shutdown_state_backend: Callable[[], Awaitable[None]],
        set_pipeline_meta: Callable[[Optional[str], str], None],
        reset_pipeline_cache: Callable[[], None],
    ) -> None:
        self.pipeline = pipeline
        self.pipeline_name = pipeline_name
        self.pipeline_version = pipeline_version
        self.pipeline_id = pipeline_id
        self.context_model = context_model
        self.initial_context_data = initial_context_data
        self.resources = resources
        self.usage_limits = usage_limits
        self.hooks = hooks
        self.backend = backend
        self.state_backend = state_backend
        self.delete_on_completion = delete_on_completion
        self._trace_manager = trace_manager
        self._ensure_pipeline_cb = ensure_pipeline
        self._refresh_pipeline_meta = refresh_pipeline_meta
        self._dispatch_hook_cb = dispatch_hook
        self._shutdown_state_backend_cb = shutdown_state_backend
        self._set_pipeline_meta = set_pipeline_meta
        self._reset_pipeline_cache = reset_pipeline_cache
        self._has_inline_pipeline = pipeline is not None

    def _ensure_pipeline(self) -> Pipeline[RunnerInT, RunnerOutT]:
        pipeline = self._ensure_pipeline_cb()
        self.pipeline = pipeline
        try:
            name, version = self._refresh_pipeline_meta()
            self.pipeline_name = name
            self.pipeline_version = version
        except Exception:
            pass
        return pipeline

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
        await self._dispatch_hook_cb(event_name, **kwargs)

    async def _shutdown_state_backend(self) -> None:
        await self._shutdown_state_backend_cb()

    async def execute_steps(
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
        """Execute pipeline steps using the execution manager."""
        assert self.pipeline is not None

        state_manager: StateManager[ContextT] = StateManager(state_backend)
        step_coordinator: StepCoordinator[ContextT] = StepCoordinator(self.hooks, self.resources)
        root_quota = build_root_quota(self.usage_limits)

        execution_manager = ExecutionManager(
            self.pipeline,
            backend=self.backend,
            state_manager=state_manager,
            usage_limits=self.usage_limits,
            step_coordinator=step_coordinator,
            root_quota=root_quota,
        )

        exec_iter = execution_manager.execute_steps(
            start_idx=start_idx,
            data=data,
            context=context,
            result=result,
            stream_last=stream_last,
            run_id=run_id,
            state_created_at=state_created_at,
        )
        try:
            async for item in exec_iter:
                yield item
        finally:
            await aclose_if_possible(exec_iter)

    async def run_async(
        self,
        initial_input: RunnerInT,
        *,
        run_id: str | None = None,
        initial_context_data: Optional[JSONObject] = None,
    ) -> AsyncIterator[object]:
        """Internal implementation for async pipeline execution."""

        async def _run_generator() -> AsyncIterator[object]:
            try:
                if get_settings().warn_legacy:
                    warnings.warn(
                        "Legacy runner path (run_async yielding PipelineResult) in use; prefer run_outcomes_async.",
                        DeprecationWarning,
                    )
            except Exception:
                pass

            try:
                from flujo.infra import telemetry

                telemetry.logfire.debug(
                    f"Runner.run_async received initial_context_data keys={list(initial_context_data.keys()) if isinstance(initial_context_data, dict) else None}"
                )
            except Exception:
                pass

            # Validate scratchpad removal once per run (without enforcing unrelated validation rules).
            try:
                validate_fn = getattr(self.pipeline, "validate", None)
                if callable(validate_fn):
                    report = validate_fn(raise_on_error=False, include_imports=True)
                    findings = []
                    try:
                        for f in getattr(report, "errors", []) or []:
                            rid = str(getattr(f, "rule_id", "") or "").upper()
                            msg = str(getattr(f, "message", "") or "")
                            if "SCRATCHPAD" in rid or "scratchpad" in msg.lower():
                                findings.append(msg or rid)
                    except Exception:
                        findings = []
                    if findings:
                        raise ConfigurationError("; ".join(findings[:3]))
            except ConfigurationError:
                raise
            except Exception:
                pass
            current_context_instance: Optional[ContextT] = None
            if self.context_model is not None:
                try:
                    context_data = (
                        copy.deepcopy(self.initial_context_data)
                        if self.initial_context_data
                        else {}
                    )
                    if initial_context_data:
                        context_data.update(copy.deepcopy(initial_context_data))
                    if run_id is not None:
                        context_data["run_id"] = run_id

                    from flujo.utils.serialization import lookup_custom_deserializer

                    processed_context_data = {}
                    for key, value in context_data.items():
                        if key in self.context_model.model_fields:
                            field_info = self.context_model.model_fields[key]
                            field_type = field_info.annotation
                            if field_type is not None and isinstance(value, dict):
                                custom_deserializer = lookup_custom_deserializer(field_type)
                                if custom_deserializer:
                                    try:
                                        reconstructed_value = custom_deserializer(value)
                                        processed_context_data[key] = reconstructed_value
                                        continue
                                    except Exception:
                                        pass
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
                current_context_instance = self._as_context_t(
                    PipelineContext(initial_prompt=str(initial_input))
                )
                try:
                    merged_data: JSONObject = {}
                    if isinstance(self.initial_context_data, dict):
                        merged_data.update(copy.deepcopy(self.initial_context_data))
                    if isinstance(initial_context_data, dict):
                        merged_data.update(copy.deepcopy(initial_context_data))
                    if "scratchpad" in merged_data:
                        raise PipelineContextInitializationError(
                            "scratchpad has been removed; migrate initial context data to typed fields "
                            "(status, step_outputs, import_artifacts, etc.)."
                        )
                    for key, value in merged_data.items():
                        if key in ("initial_prompt", "run_id"):
                            object.__setattr__(current_context_instance, key, value)
                        else:
                            object.__setattr__(current_context_instance, key, value)
                except PipelineContextInitializationError:
                    raise
                except Exception:
                    pass
                if run_id is not None:
                    object.__setattr__(current_context_instance, "run_id", run_id)

            if hasattr(current_context_instance, "__dict__"):
                if not hasattr(current_context_instance, "_artifacts"):
                    object.__setattr__(current_context_instance, "_artifacts", [])

            if isinstance(current_context_instance, PipelineContext):
                current_context_instance.status = "running"

            start_idx = 0
            data: object = initial_input
            pipeline_result_obj: PipelineResult[ContextT] = PipelineResult()
            state_created_at: datetime | None = None
            state_manager: StateManager[ContextT] = StateManager(self.state_backend)
            run_id_for_state = run_id or getattr(current_context_instance, "run_id", None)

            if run_id_for_state is None:
                run_id_for_state = state_manager.get_run_id_from_context(
                    self._as_context_t(current_context_instance)
                )

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
                    current_context_instance = context
                    start_idx = max(start_idx, current_idx)
                    state_created_at = created_at
                    if current_idx > 0:
                        data = last_output
                        # When resuming, keep existing history but do not overwrite if already populated
                        if not pipeline_result_obj.step_history:
                            pipeline_result_obj.step_history = step_history

                    if pipeline_version is not None:
                        self.pipeline_version = pipeline_version
                        self._set_pipeline_meta(self.pipeline_name, pipeline_version)
                    if pipeline_name is not None:
                        self.pipeline_name = pipeline_name
                        self._set_pipeline_meta(
                            pipeline_name, pipeline_version or self.pipeline_version
                        )
                    # Invalidate cached pipeline so resolver reloads the persisted version
                    if not self._has_inline_pipeline:
                        self._reset_pipeline_cache()

                    self._ensure_pipeline()

                    assert self.pipeline is not None
                    if start_idx > len(self.pipeline.steps):
                        raise ExecutionError(
                            f"Invalid persisted step index {start_idx} for pipeline with {len(self.pipeline.steps)} steps"
                        )
                else:
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

                try:
                    self._ensure_pipeline()
                    _num_steps = len(self.pipeline.steps) if self.pipeline is not None else 0
                except Exception:
                    _num_steps = 0
                if _num_steps > 1:
                    await state_manager.persist_workflow_state_optimized(
                        run_id=run_id_for_state,
                        context=self._as_context_t(current_context_instance),
                        current_step_index=start_idx,
                        last_step_output=data,
                        status="running",
                        state_created_at=state_created_at,
                    )
            else:
                self._ensure_pipeline()
            cancelled = False
            paused = False
            pipeline = self._ensure_pipeline()
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

                def _has_missing_terminal_outcome(step_result: object) -> bool:
                    fb = getattr(step_result, "feedback", None)
                    return isinstance(fb, str) and "no terminal outcome" in fb.lower()

                def _prepare_pipeline_result_for_emit(
                    *, pad_missing: bool = True
                ) -> PipelineResult[ContextT]:
                    nonlocal pipeline_result_obj
                    if pad_missing:
                        expected_steps = len(pipeline.steps)
                        have = len(pipeline_result_obj.step_history)
                        if expected_steps > 0 and have < expected_steps and start_idx == 0:
                            missing_outcome_detected = any(
                                _has_missing_terminal_outcome(sr)
                                for sr in pipeline_result_obj.step_history
                            ) or all(
                                getattr(sr, "success", True)
                                for sr in pipeline_result_obj.step_history
                            )
                            if missing_outcome_detected:
                                from flujo.domain.models import StepResult as _SR

                                for j in range(have, expected_steps):
                                    try:
                                        missing_name = getattr(
                                            pipeline.steps[j], "name", f"step_{j}"
                                        )
                                    except Exception:
                                        missing_name = f"step_{j}"
                                    if any(
                                        getattr(sr, "name", None) == str(missing_name)
                                        for sr in pipeline_result_obj.step_history
                                    ):
                                        continue
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
                    return pipeline_result_obj

                step_iter = self.execute_steps(
                    start_idx,
                    data,
                    self._as_context_t(current_context_instance),
                    pipeline_result_obj,
                    stream_last=True,
                    run_id=run_id_for_state,
                    state_backend=self.state_backend,
                    state_created_at=state_created_at,
                )
                try:
                    async for chunk in step_iter:
                        if isinstance(chunk, PipelineResult):
                            expected_steps = len(pipeline.steps)
                            have = len(chunk.step_history)
                            is_paused = False
                            try:
                                ctx = chunk.final_pipeline_context
                                if ctx and getattr(ctx, "status", None) == "paused":
                                    is_paused = True
                            except Exception:
                                pass

                            if (
                                not is_paused
                                and expected_steps > 0
                                and have < expected_steps
                                and start_idx == 0
                            ):
                                missing_outcome_detected = any(
                                    _has_missing_terminal_outcome(sr) for sr in chunk.step_history
                                ) or all(getattr(sr, "success", True) for sr in chunk.step_history)
                                if missing_outcome_detected:
                                    from flujo.domain.models import StepResult as _SR

                                    for j in range(have, expected_steps):
                                        try:
                                            missing_name = getattr(
                                                pipeline.steps[j], "name", f"step_{j}"
                                            )
                                        except Exception:
                                            missing_name = f"step_{j}"
                                        if any(
                                            getattr(sr, "name", None) == str(missing_name)
                                            for sr in chunk.step_history
                                        ):
                                            continue
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
                            try:
                                ctx = chunk.final_pipeline_context
                                if (
                                    isinstance(ctx, PipelineContext)
                                    and getattr(ctx, "status", None) == "paused"
                                ):
                                    paused = True
                                    pipeline_result_obj = chunk
                                    _yielded_pipeline_result = True
                                    try:
                                        chunk.success = False
                                    except Exception:
                                        pass
                                    yield chunk
                                    return
                                if isinstance(ctx, PipelineContext) and (
                                    getattr(ctx, "loop_iteration_index", None) is not None
                                    or getattr(ctx, "loop_step_index", None) is not None
                                ):
                                    try:
                                        ctx.status = "paused"
                                        pipeline_result_obj = chunk
                                        paused = True
                                        _yielded_pipeline_result = True
                                        chunk.success = False
                                        yield chunk
                                        return
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            pipeline_result_obj = chunk
                            try:
                                if chunk.final_pipeline_context is not None:
                                    current_context_instance = self._as_context_t(
                                        chunk.final_pipeline_context
                                    )
                            except Exception:
                                pass
                            _yielded_pipeline_result = True
                        yield chunk
                finally:
                    await aclose_if_possible(step_iter)
                if not _yielded_pipeline_result:
                    _yielded_pipeline_result = True
                    yield _prepare_pipeline_result_for_emit()
            except asyncio.CancelledError:
                telemetry.logfire.info("Pipeline cancelled")
                cancelled = True
                try:
                    _yielded_pipeline_result = True
                    yield _prepare_pipeline_result_for_emit(pad_missing=False)
                except Exception:
                    pass
                return
            except PipelineAbortSignal as e:
                telemetry.logfire.debug(str(e))
                paused = True
                try:
                    if isinstance(current_context_instance, PipelineContext):
                        current_context_instance.status = "paused"
                        if current_context_instance.pause_message is None:
                            current_context_instance.pause_message = str(e) or "Paused for HITL"
                        pipeline_result_obj.final_pipeline_context = self._as_context_t(
                            current_context_instance
                        )
                except Exception:
                    pass
                if not _yielded_pipeline_result:
                    _yielded_pipeline_result = True
                    yield _prepare_pipeline_result_for_emit(pad_missing=False)
            except (UsageLimitExceededError, PricingNotConfiguredError) as e:
                if current_context_instance is not None:
                    assert self.pipeline is not None
                    execution_manager: ExecutionManager[ContextT] = ExecutionManager(self.pipeline)
                    if pipeline_result_obj.final_pipeline_context is None:
                        execution_manager.set_final_context(
                            pipeline_result_obj,
                            self._as_context_t(current_context_instance),
                        )
                    if isinstance(e, UsageLimitExceededError):
                        if e.result is None:
                            e.result = pipeline_result_obj
                        else:
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
                    pipeline_result_obj.trace_tree = getattr(
                        self._trace_manager, "_root_span", None
                    )
                if current_context_instance is not None:
                    assert self.pipeline is not None
                    exec_manager = ExecutionManager(
                        self.pipeline,
                        state_manager=state_manager,
                    )
                    if pipeline_result_obj.final_pipeline_context is None:
                        exec_manager.set_final_context(
                            pipeline_result_obj,
                            self._as_context_t(current_context_instance),
                        )
                    # If we resumed and did not execute any new steps (empty history), force paused so HITL can continue
                    if start_idx > 0 and not pipeline_result_obj.step_history:
                        paused = True
                        try:
                            if isinstance(current_context_instance, PipelineContext):
                                current_context_instance.status = "paused"
                        except Exception:
                            pass
                    final_status: Literal[
                        "running", "paused", "completed", "failed", "cancelled"
                    ] = "failed"
                    # Resume semantics: keep paused unless we actually executed remaining steps
                    if cancelled:
                        final_status = "failed"
                    elif paused or (
                        isinstance(current_context_instance, PipelineContext)
                        and getattr(current_context_instance, "status", None) == "paused"
                    ):
                        final_status = "paused"
                    elif start_idx > 0:
                        expected_steps = len(pipeline.steps)
                        executed = len(pipeline_result_obj.step_history)
                        executed_success = all(s.success for s in pipeline_result_obj.step_history)
                        # On resume, require full coverage to call it completed
                        if executed_success and executed >= expected_steps:
                            final_status = "completed"
                        else:
                            final_status = "paused"
                    elif pipeline_result_obj.step_history:
                        expected: int | None = len(pipeline.steps)
                        executed_success = all(s.success for s in pipeline_result_obj.step_history)
                        if (
                            expected is not None
                            and len(pipeline_result_obj.step_history) == expected
                            and executed_success
                        ):
                            final_status = "completed"
                        else:
                            final_status = "failed"
                    else:
                        num_steps = len(pipeline.steps)
                        if num_steps == 0:
                            final_status = "completed"

                    await exec_manager.persist_final_state(
                        run_id=run_id_for_state,
                        context=self._as_context_t(current_context_instance),
                        result=pipeline_result_obj,
                        start_idx=start_idx,
                        state_created_at=state_created_at,
                        final_status=final_status,
                    )
                    try:
                        # Require full pipeline coverage to mark success on resume.
                        expected_len = len(pipeline.steps)
                        pipeline_result_obj.success = (
                            final_status == "completed"
                            and len(pipeline_result_obj.step_history) >= expected_len
                        )
                    except Exception:
                        pass
                    if (
                        self.delete_on_completion
                        and final_status == "completed"
                        and run_id_for_state is not None
                    ):
                        await state_manager.delete_workflow_state(run_id_for_state)
                        try:
                            if self.state_backend is not None:
                                await self.state_backend.delete_state(run_id_for_state)
                        except Exception:
                            pass
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
                    telemetry.logfire.info("Skipping post_run hook due to cancellation")
                except PipelineAbortSignal as e:
                    telemetry.logfire.debug(str(e))

            return

        gen = _run_generator()
        try:
            async for item in gen:
                yield item
        finally:
            await aclose_if_possible(gen)
            await self._shutdown_state_backend()
