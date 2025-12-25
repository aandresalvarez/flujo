from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable, TypeVar, Union

from pydantic import BaseModel as PydanticBaseModel
from ....domain.models import BaseModel

from ....exceptions import UsageLimitExceededError
from ....domain.models import UsageLimits, Quota
from ....domain.sandbox import SandboxProtocol
from ..executor_protocols import IHasher, ISerializer
from ..context_manager import ContextManager
from ..types import ExecutionFrame
from ....domain.dsl.cache_step import CacheStep
from ....domain.dsl.loop import LoopStep
from ....domain.dsl.parallel import ParallelStep
from ....domain.dsl.conditional import ConditionalStep
from ....domain.dsl.dynamic_router import DynamicParallelRouterStep
from ....domain.dsl.step import HumanInTheLoopStep, Step
from ....exceptions import (
    InfiniteFallbackError,
    MissingAgentError,
    PausedException,
    PipelineAbortSignal,
    InfiniteRedirectError,
)
from ....domain.models import PipelineResult, StepResult, StepOutcome, Failure, Success
from ..failure_builder import build_failure_outcome
from ..context.context_vars import _CACHE_OVERRIDE

__all__ = ["_CACHE_OVERRIDE"]  # Re-export for backward compatibility

Outcome = Union[StepOutcome[StepResult], StepResult]

TCtx = TypeVar("TCtx", bound=BaseModel)


# Backward-compatible error types used by other modules
class RetryableError(Exception):
    """Base class for errors that should trigger retries."""

    pass


class ValidationError(RetryableError):
    """Validation failures that can be retried."""

    pass


class PluginError(RetryableError):
    """Plugin failures that can be retried."""

    pass


class AgentError(RetryableError):
    """Agent execution errors that can be retried."""

    pass


class ContextIsolationError(Exception):
    """Raised when context isolation fails under strict settings."""

    pass


class ContextMergeError(Exception):
    """Raised when context merging fails under strict settings."""

    pass


@dataclass
class _Frame:
    """Frame class for backward compatibility with tests."""

    step: object
    data: object
    context: object | None = None
    resources: object | None = None


StepExecutor = Callable[
    [Step[object, object], object, object | None, object | None, object | None],
    Awaitable[StepResult],
]


def safe_step_name(step: object) -> str:
    """Return a best-effort, mock-tolerant step name for logging/telemetry."""
    try:
        if hasattr(step, "name"):
            name = step.name
            if hasattr(name, "_mock_name"):
                if hasattr(name, "_mock_return_value") and name._mock_return_value:
                    return str(name._mock_return_value)
                if hasattr(name, "_mock_name") and name._mock_name:
                    return str(name._mock_name)
                return "mock_step"
            return str(name)
        return "unknown_step"
    except Exception:
        return "unknown_step"


def format_feedback(
    feedback: Optional[str], default_message: str = "Agent execution failed"
) -> str:
    """Normalize feedback strings to include an error prefix when absent."""
    if feedback is None or str(feedback).strip() == "":
        return default_message
    msg = str(feedback)
    low = msg.lower()
    if ("error" not in low) and ("exception" not in low):
        base = default_message or "Agent exception"
        if ("error" not in base.lower()) and ("exception" not in base.lower()):
            base = "Agent exception"
        return f"{base}: {msg}"
    return msg


def normalize_frame_context(frame: object) -> None:
    """Retain context/resources/limits on the frame for policy access."""
    # Accessors kept for completeness; policies pull from frame directly
    _ = getattr(frame, "context", None)
    _ = getattr(frame, "resources", None)
    _ = getattr(frame, "limits", None)
    _ = getattr(frame, "on_chunk", None)


def enforce_typed_context(context: object | None) -> BaseModel | None:
    """Enforce that context is a Pydantic BaseModel when strict mode is enabled.

    Strict mode rejects legacy dict contexts instead of attempting coercion, to
    avoid silent shape drift and to keep context contracts explicit.

    Note: Accepts any pydantic.BaseModel subclass, not just flujo.domain.models.BaseModel,
    to support user-defined context models that don't inherit from flujo's base classes.
    At runtime, plain pydantic.BaseModel instances are accepted; the return type uses
    flujo's BaseModel for internal type system compatibility.
    """
    if context is None:
        return None
    # Accept any pydantic BaseModel subclass (including flujo's BaseModel)
    if isinstance(context, PydanticBaseModel):
        # Cast to flujo's BaseModel for type system compatibility with ExecutionFrame
        return context  # type: ignore[return-value]

    # Strict-only posture: no opt-out for non-Pydantic contexts.
    raise TypeError("Context must be a Pydantic BaseModel (strict mode enforced).")


def _coerce_context_model(context: Optional[TCtx]) -> Optional[TCtx]:
    """Validate and narrow incoming contexts to typed Pydantic models."""
    if context is None:
        return None
    # Accept any pydantic BaseModel subclass (including flujo's BaseModel)
    if not isinstance(context, PydanticBaseModel):
        raise TypeError("Context must be a Pydantic BaseModel (strict mode enforced).")
    return context


def _ensure_same_context_type(value: BaseModel, reference: TCtx) -> TCtx:
    """Ensure ContextManager returns the same context model type as provided."""
    if not isinstance(value, type(reference)):
        raise TypeError(
            f"ContextManager returned {type(value).__name__}, expected {type(reference).__name__}"
        )
    return value


def _ensure_outcome(value: object, *, step_name: str = "") -> Outcome:
    """Runtime guard to enforce Outcome shape and avoid unsafe casts."""
    if isinstance(value, StepOutcome):
        return value
    if isinstance(value, StepResult):
        return value
    raise TypeError(
        f"Expected StepOutcome or StepResult for step '{step_name or '<unknown>'}', "
        f"got {type(value).__name__}"
    )


def attach_sandbox_to_context(context: object | None, sandbox: SandboxProtocol | None) -> None:
    """Attach sandbox handle to context when present without mutating dict contexts."""
    if context is None or sandbox is None:
        return
    if isinstance(context, dict):
        return
    try:
        existing = getattr(context, "sandbox", None)
        if existing is not None:
            return
    except Exception:
        pass
    try:
        existing = getattr(context, "_sandbox", None)
        if existing is not None:
            return
    except Exception:
        pass
    try:
        object.__setattr__(context, "_sandbox", sandbox)
        return
    except Exception:
        pass
    try:
        setattr(context, "sandbox", sandbox)
    except Exception:
        pass


def attach_memory_store_to_context(context: object | None, store: object | None) -> None:
    """Attach memory store handle to context when present without mutating dict contexts."""
    if context is None or store is None:
        return
    if isinstance(context, dict):
        return
    for attr in ("memory_store", "_memory_store"):
        try:
            existing = getattr(context, attr)
            if existing is not None:
                return
        except Exception:
            continue
    try:
        object.__setattr__(context, "_memory_store", store)
        return
    except Exception:
        pass
    try:
        setattr(context, "memory_store", store)
    except Exception:
        pass


async def set_quota_and_hydrate(
    frame: object, quota_manager: object, hydration_manager: object
) -> None:
    """Assign quota to the execution context and hydrate managed state."""
    try:
        setter = getattr(quota_manager, "set_current_quota", None)
        if callable(setter):
            setter(getattr(frame, "quota", None))
    except Exception:
        pass
    try:
        hydrate = getattr(hydration_manager, "hydrate_context", None)
        if callable(hydrate):
            await hydrate(getattr(frame, "context", None))
    except Exception:
        pass


def get_current_quota(quota_manager: object) -> Optional[Quota]:
    """Best-effort getter for the current quota using the manager first."""
    try:
        getter = getattr(quota_manager, "get_current_quota", None)
        quota = getter() if callable(getter) else None
        return quota if isinstance(quota, Quota) else None
    except Exception:
        pass
    return None


def set_current_quota(quota_manager: object, quota: Optional[Quota]) -> object | None:
    """Best-effort setter for the current quota (returns token when available)."""
    try:
        setter = getattr(quota_manager, "set_current_quota", None)
        return setter(quota) if callable(setter) else None
    except Exception:
        return None


def reset_current_quota(quota_manager: object, token: Optional[object]) -> None:
    """Best-effort reset for quota context tokens."""
    try:
        if token is not None and hasattr(token, "old_value"):
            setter = getattr(quota_manager, "set_current_quota", None)
            if callable(setter):
                setter(token.old_value)
            return
    except Exception:
        pass


def hash_obj(obj: object | None, serializer: ISerializer, hasher: IHasher) -> str:
    """Hash arbitrary objects using provided serializer/hasher."""
    if obj is None:
        return "None"
    if isinstance(obj, bytes):
        return hasher.digest(obj)
    if isinstance(obj, str):
        return hasher.digest(obj.encode("utf-8"))
    try:
        serialized = serializer.serialize(obj)
        return hasher.digest(serialized)
    except Exception:
        return hasher.digest(str(obj).encode("utf-8"))


async def maybe_launch_background(core: object, frame: ExecutionFrame[TCtx]) -> object | None:
    """Launch background step if applicable and return outcome."""
    try:
        bg_mgr = getattr(core, "_background_task_manager", None)
        if bg_mgr is None:
            return None
        launch = getattr(bg_mgr, "maybe_launch_background_step", None)
        if not callable(launch):
            return None
        bg_outcome = await launch(core=core, frame=frame)
        if bg_outcome is None:
            return None
        if not isinstance(bg_outcome, (StepOutcome, StepResult)):
            raise TypeError(
                f"Background task manager returned unsupported type {type(bg_outcome).__name__}"
            )
        return bg_outcome
    except Exception:
        return None


def log_step_start(core: object, step: object, *, stream: bool, fallback_depth: int) -> None:
    """Delegate telemetry logging for step start."""
    try:
        telemetry_handler = getattr(core, "_telemetry_handler", None)
        log_fn = getattr(telemetry_handler, "log_step_start", None)
        if callable(log_fn):
            log_fn(step, stream=stream, fallback_depth=fallback_depth)
    except Exception:
        return


def log_execution_error(core: object, step_name: str, exc: Exception) -> None:
    """Delegate telemetry logging for execution errors."""
    try:
        telemetry_handler = getattr(core, "_telemetry_handler", None)
        log_fn = getattr(telemetry_handler, "log_execution_error", None)
        if callable(log_fn):
            log_fn(step_name, exc)
    except Exception:
        return


async def execute_flow(
    core: object,
    frame: ExecutionFrame[TCtx],
    called_with_frame: bool,
) -> StepOutcome[StepResult] | StepResult:
    """Execute a frame through cache/dispatch/persist pipeline."""
    set_quota_and_hydrate_fn = getattr(core, "_set_quota_and_hydrate", None)
    if callable(set_quota_and_hydrate_fn):
        await set_quota_and_hydrate_fn(frame)

    step = frame.step

    bg_outcome = await maybe_launch_background(core, frame)
    if bg_outcome is not None:
        return _ensure_outcome(bg_outcome, step_name=getattr(step, "name", ""))

    normalize_frame_context_fn = getattr(core, "_normalize_frame_context", None)
    if callable(normalize_frame_context_fn):
        normalize_frame_context_fn(frame)
    stream = getattr(frame, "stream", False)
    _fallback_depth = getattr(frame, "_fallback_depth", 0)

    log_step_start(core, step, stream=stream, fallback_depth=_fallback_depth)

    maybe_use_cache_fn = getattr(core, "_maybe_use_cache", None)
    if not callable(maybe_use_cache_fn):
        raise TypeError("ExecutorCore missing _maybe_use_cache")
    cached_outcome, cache_key = await maybe_use_cache_fn(frame, called_with_frame=called_with_frame)
    frame.cache_checked = True
    if cached_outcome is not None:
        return _ensure_outcome(cached_outcome, step_name=getattr(step, "name", ""))

    try:
        dispatch_fn = getattr(core, "_dispatch_frame", None)
        if not callable(dispatch_fn):
            raise TypeError("ExecutorCore missing _dispatch_frame")
        result_outcome = _ensure_outcome(
            await dispatch_fn(frame, called_with_frame=called_with_frame),
            step_name=getattr(step, "name", ""),
        )
    except asyncio.CancelledError:
        # Preserve cancellation semantics for signal handling tests and callers.
        raise
    except MissingAgentError as e:
        handle_missing_agent_fn = getattr(core, "_handle_missing_agent_exception", None)
        if not callable(handle_missing_agent_fn):
            raise
        handled = handle_missing_agent_fn(e, step, called_with_frame=called_with_frame)
        if handled is not None:
            return _ensure_outcome(handled, step_name=getattr(step, "name", ""))
        raise
    except UsageLimitExceededError:
        raise
    except InfiniteFallbackError:
        # Control-flow exceptions must propagate to allow orchestrators to react.
        raise
    except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
        raise
    except Exception as exc:
        unexpected_fn = getattr(core, "_handle_unexpected_exception", None)
        if not callable(unexpected_fn):
            raise
        return _ensure_outcome(
            unexpected_fn(step=step, frame=frame, exc=exc, called_with_frame=called_with_frame),
            step_name=getattr(step, "name", ""),
        )

    if isinstance(result_outcome, StepOutcome):
        return result_outcome
    result = result_outcome

    persist_fn = getattr(core, "_persist_and_finalize", None)
    if not callable(persist_fn):
        raise TypeError("ExecutorCore missing _persist_and_finalize")
    return _ensure_outcome(
        await persist_fn(
            step=step,
            result=result,
            cache_key=cache_key,
            called_with_frame=called_with_frame,
            frame=frame if called_with_frame else None,
        ),
        step_name=getattr(step, "name", ""),
    )


def build_failure(
    core: object,
    *,
    step: object,
    frame: ExecutionFrame[TCtx],
    exc: Exception,
    called_with_frame: bool,
) -> Failure[StepResult]:
    """Delegate failure outcome construction."""
    safe_step_name_fn = getattr(core, "_safe_step_name", safe_step_name)
    return build_failure_outcome(
        step=step,
        frame=frame,
        exc=exc,
        called_with_frame=called_with_frame,
        safe_step_name=safe_step_name_fn,
    )


def handle_missing_agent_exception(
    core: object,
    err: MissingAgentError,
    step: object,
    *,
    called_with_frame: bool,
) -> StepOutcome[StepResult] | StepResult:
    """Delegate missing agent handling."""
    result_handler = getattr(core, "_result_handler", None)
    if result_handler is None:
        raise TypeError("ExecutorCore missing _result_handler")
    handle_fn = getattr(result_handler, "handle_missing_agent_exception", None)
    if not callable(handle_fn):
        raise TypeError("ExecutorCore missing ResultHandler.handle_missing_agent_exception")
    handled = handle_fn(err, step, called_with_frame=called_with_frame)
    return _ensure_outcome(handled, step_name=getattr(step, "name", ""))


async def persist_and_finalize(
    core: object,
    *,
    step: object,
    result: StepResult,
    cache_key: Optional[str],
    called_with_frame: bool,
    frame: ExecutionFrame[TCtx] | None = None,
) -> StepOutcome[StepResult] | StepResult:
    """Delegate cache persist/finalize."""
    result_handler = getattr(core, "_result_handler", None)
    if result_handler is None:
        raise TypeError("ExecutorCore missing _result_handler")
    persist_fn = getattr(result_handler, "persist_and_finalize", None)
    if not callable(persist_fn):
        raise TypeError("ExecutorCore missing ResultHandler.persist_and_finalize")
    finalized = await persist_fn(
        step=step,
        result=result,
        cache_key=cache_key,
        called_with_frame=called_with_frame,
        frame=frame,
    )
    return _ensure_outcome(finalized, step_name=getattr(step, "name", ""))


def handle_unexpected_exception(
    core: object,
    *,
    step: object,
    frame: ExecutionFrame[TCtx],
    exc: Exception,
    called_with_frame: bool,
) -> StepOutcome[StepResult] | StepResult:
    """Delegate unexpected exception handling."""
    result_handler = getattr(core, "_result_handler", None)
    if result_handler is None:
        raise TypeError("ExecutorCore missing _result_handler")
    handle_fn = getattr(result_handler, "handle_unexpected_exception", None)
    if not callable(handle_fn):
        raise TypeError("ExecutorCore missing ResultHandler.handle_unexpected_exception")
    handled = handle_fn(step=step, frame=frame, exc=exc, called_with_frame=called_with_frame)
    return _ensure_outcome(handled, step_name=getattr(step, "name", ""))


async def maybe_use_cache(
    core: object,
    frame: ExecutionFrame[TCtx],
    *,
    called_with_frame: bool,
) -> tuple[Optional[StepOutcome[StepResult] | StepResult], Optional[str]]:
    """Delegate cache hit retrieval."""
    result_handler = getattr(core, "_result_handler", None)
    if result_handler is None:
        raise TypeError("ExecutorCore missing _result_handler")
    cache_fn = getattr(result_handler, "maybe_use_cache", None)
    if not callable(cache_fn):
        raise TypeError("ExecutorCore missing ResultHandler.maybe_use_cache")
    cached_outcome, cache_key = await cache_fn(frame, called_with_frame=called_with_frame)
    if cached_outcome is None:
        return None, cache_key
    return _ensure_outcome(cached_outcome, step_name=getattr(frame.step, "name", "")), cache_key


async def execute_entrypoint(
    core: object,
    frame_or_step: object | None = None,
    data: object | None = None,
    **kwargs: object,
) -> StepOutcome[StepResult] | StepResult:
    """Entrypoint for ExecutorCore.execute (frame or step signature)."""
    if isinstance(frame_or_step, ExecutionFrame):
        called_with_frame = True
        frame = frame_or_step
    else:
        called_with_frame = False
        allowed_keys = {
            "step",
            "data",
            "context",
            "resources",
            "limits",
            "context_setter",
            "stream",
            "on_chunk",
            "_fallback_depth",
            "quota",
            "result",
            "usage_limits",
            # Backward compatibility: cache-key overrides may be passed by legacy callers/tests
            "cache_key",
        }
        unknown_keys = set(kwargs).difference(allowed_keys)
        if unknown_keys:
            raise TypeError(
                f"Unsupported ExecutorCore.execute() arguments: {', '.join(sorted(unknown_keys))}"
            )
        # New run: reset per-run history to avoid unbounded growth across executions.
        try:
            step_history_tracker = getattr(core, "_step_history_tracker", None)
            clear_fn = getattr(step_history_tracker, "clear_history", None)
            if callable(clear_fn):
                clear_fn()
        except Exception:
            pass
        step_obj = frame_or_step if frame_or_step is not None else kwargs.get("step")
        if step_obj is None:
            raise ValueError("ExecutorCore.execute requires a Step or ExecutionFrame")
        payload = data if data is not None else kwargs.get("data")
        fb_depth_raw = kwargs.get("_fallback_depth", 0)
        fb_depth_norm = 0
        if isinstance(fb_depth_raw, (int, float, str)):
            try:
                fb_depth_norm = int(fb_depth_raw)
            except Exception:
                fb_depth_norm = 0

        context_raw = kwargs.get("context")
        context: BaseModel | None
        if isinstance(context_raw, dict):
            # Backward compatibility for legacy ExecutorCore.execute(..., context={...}).
            # `PipelineContext` rejects removed/unsafe legacy fields (e.g. scratchpad).
            from ....domain.models import PipelineContext

            context = PipelineContext.model_validate(context_raw)
        else:
            context = enforce_typed_context(context_raw)

        resources = kwargs.get("resources")

        limits_raw = kwargs.get("limits")
        limits = limits_raw if isinstance(limits_raw, UsageLimits) else None
        usage_limits_raw = kwargs.get("usage_limits")
        if limits is None and isinstance(usage_limits_raw, UsageLimits):
            limits = usage_limits_raw

        context_setter_raw = kwargs.get("context_setter")
        context_setter: Callable[[PipelineResult[BaseModel], BaseModel | None], None] | None = None
        if callable(context_setter_raw):
            context_setter_fn = context_setter_raw

            def _context_setter(res: PipelineResult[BaseModel], ctx: BaseModel | None) -> None:
                try:
                    context_setter_fn(res, ctx)
                except TypeError:
                    context_setter_fn(res)

            context_setter = _context_setter

        stream = bool(kwargs.get("stream", False))

        on_chunk_raw = kwargs.get("on_chunk")
        on_chunk: Callable[[object], Awaitable[None]] | None = None
        if callable(on_chunk_raw):
            import inspect

            on_chunk_fn = on_chunk_raw

            async def _on_chunk(chunk: object) -> None:
                res = on_chunk_fn(chunk)
                if inspect.isawaitable(res):
                    await res

            on_chunk = _on_chunk

        quota_raw = kwargs.get("quota")
        quota = quota_raw if isinstance(quota_raw, Quota) else None

        result_raw = kwargs.get("result")
        result = result_raw if isinstance(result_raw, StepResult) else None
        frame = make_execution_frame(
            core,
            step_obj,
            payload,
            context,
            resources,
            limits,
            context_setter,
            stream=stream,
            on_chunk=on_chunk,
            fallback_depth=fb_depth_norm,
            quota=quota,
            result=result,
        )

    return await execute_flow(core, frame, called_with_frame)


async def run_validation(
    core: object,
    *,
    step: Step[object, object],
    output: object,
    context: object | None,
    limits: object | None,
    data: object,
    attempt_context: object | None,
    attempt_resources: object | None,
    stream: bool,
    on_chunk: object | None,
    fallback_depth: int,
) -> StepOutcome[StepResult] | None:
    """Centralized validation + fallback handling."""
    validation_orchestrator = getattr(core, "_validation_orchestrator", None)
    validate_fn = getattr(validation_orchestrator, "validate", None)
    if not callable(validate_fn):
        raise TypeError("ExecutorCore missing _validation_orchestrator.validate")
    validation_raw = await validate_fn(
        core=core,
        step=step,
        output=output,
        context=context,
        limits=limits,
        data=data,
        attempt_context=attempt_context,
        attempt_resources=attempt_resources,
        stream=stream,
        on_chunk=on_chunk,
        fallback_depth=fallback_depth,
    )
    validation_result: Optional[Union[StepOutcome[StepResult], StepResult]]
    if validation_raw is None:
        validation_result = None
    elif isinstance(validation_raw, (StepOutcome, StepResult)):
        validation_result = validation_raw
    else:
        raise TypeError(
            f"Validation orchestrator returned unsupported type {type(validation_raw).__name__}"
        )
    if validation_result is None:
        return None
    if isinstance(validation_result, StepResult) and validation_result.success:
        return Success(step_result=validation_result)
    if isinstance(validation_result, StepOutcome):
        return validation_result
    return Failure(
        error=Exception(validation_result.feedback or "Validation failed"),
        feedback=validation_result.feedback,
        step_result=validation_result,
    )


def make_execution_frame(
    core: object,
    step: object,
    data: object,
    context: BaseModel | None,
    resources: object | None,
    limits: UsageLimits | None,
    context_setter: Callable[[PipelineResult[BaseModel], BaseModel | None], None] | None,
    *,
    stream: bool = False,
    on_chunk: Callable[[object], Awaitable[None]] | None = None,
    fallback_depth: int = 0,
    quota: Quota | None = None,
    result: StepResult | None = None,
) -> ExecutionFrame[BaseModel]:
    """Create an ExecutionFrame with the current quota context."""
    context = enforce_typed_context(context)
    sandbox = getattr(core, "sandbox", None)
    attach_sandbox_to_context(context, sandbox)
    try:
        attach_memory_store_to_context(context, getattr(core, "memory_store", None))
    except Exception:
        pass
    quota_value = quota
    if quota_value is None:
        try:
            quota_manager = getattr(core, "_quota_manager", None)
            get_quota_fn = getattr(quota_manager, "get_current_quota", None)
            quota_value = get_quota_fn() if callable(get_quota_fn) else None
        except Exception:
            quota_value = None
    return ExecutionFrame(
        step=step,
        data=data,
        context=context,
        resources=resources,
        limits=limits,
        quota=quota_value,
        stream=stream,
        on_chunk=on_chunk,
        context_setter=context_setter or (lambda _res, _ctx: None),
        result=result,
        _fallback_depth=fallback_depth,
    )


def isolate_context(context: Optional[TCtx], *, strict_context_isolation: bool) -> Optional[TCtx]:
    """Isolate context using ContextManager with strict toggle (typed)."""
    ctx_model = _coerce_context_model(context)
    if ctx_model is None:
        return None
    if strict_context_isolation:
        isolated = ContextManager.isolate_strict(ctx_model)
    else:
        isolated = ContextManager.isolate(ctx_model)
    if isolated is None:
        return None
    return _ensure_same_context_type(isolated, ctx_model)


def merge_context_updates(
    main_context: Optional[TCtx],
    branch_context: Optional[TCtx],
    *,
    strict_context_merge: bool,
) -> Optional[TCtx]:
    """Merge two contexts using ContextManager with strict toggle (typed)."""
    if main_context is None and branch_context is None:
        return None
    if main_context is None:
        return _coerce_context_model(branch_context)
    if branch_context is None:
        return _coerce_context_model(main_context)

    main_ctx = _coerce_context_model(main_context)
    branch_ctx = _coerce_context_model(branch_context)
    assert main_ctx is not None
    assert branch_ctx is not None

    if strict_context_merge:
        merged = ContextManager.merge_strict(main_ctx, branch_ctx)
    else:
        merged = ContextManager.merge(main_ctx, branch_ctx)
    if merged is None:
        return None
    return _ensure_same_context_type(merged, main_ctx)


def accumulate_loop_context(
    current_context: Optional[TCtx],
    iteration_context: Optional[TCtx],
    *,
    strict_context_merge: bool,
) -> Optional[TCtx]:
    """Merge loop iteration context into current context."""
    if current_context is None:
        return _coerce_context_model(iteration_context)
    if iteration_context is None:
        return _coerce_context_model(current_context)
    return merge_context_updates(
        current_context, iteration_context, strict_context_merge=strict_context_merge
    )


def update_context_state(context: object | None, state: str) -> None:
    """Annotate context state best-effort (typed field preferred; no scratchpad writes)."""
    if context is None:
        return
    try:
        if hasattr(context, "status"):
            # Only set when it matches our known execution states.
            if state in {"running", "paused", "completed", "failed"}:
                try:
                    context.status = state
                except Exception:
                    pass
    except Exception:
        pass


def is_complex_step(step: object) -> bool:
    """Identify complex steps for dispatch decisions."""
    try:
        if hasattr(step, "is_complex"):
            prop = getattr(step, "is_complex")
            if callable(prop):
                try:
                    if bool(prop()):
                        return True
                except Exception:
                    pass
            else:
                if bool(prop):
                    return True
    except Exception:
        pass

    if isinstance(
        step,
        (
            ParallelStep,
            LoopStep,
            ConditionalStep,
            DynamicParallelRouterStep,
            HumanInTheLoopStep,
            CacheStep,
        ),
    ):
        return True

    try:
        if getattr(step, "name", None) == "cache":
            return True
    except Exception:
        pass

    try:
        plugins = getattr(step, "plugins", None)
        if isinstance(plugins, (list, tuple)):
            if len(plugins) > 0:
                return True
        elif plugins:
            return True
    except Exception:
        pass

    try:
        if hasattr(step, "meta") and isinstance(step.meta, dict):
            if step.meta.get("is_validation_step"):
                return True
    except Exception:
        pass
    return False


@dataclass
class _UsageTracker:
    """Lightweight usage tracker retained for backward compatibility in tests."""

    total_cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _lock: Optional[asyncio.Lock] = field(init=False, default=None)

    def _get_lock(self) -> asyncio.Lock:
        """Lazily create the lock on first access."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def add(self, cost_usd: float, tokens: int) -> None:
        async with self._get_lock():
            self.total_cost_usd += float(cost_usd)
            self.prompt_tokens += int(tokens)

    async def guard(self, limits: UsageLimits) -> None:  # noqa: ARG002
        # Backward-compatibility stub: quota reservation is the only enforcement surface.
        _ = limits
        return None

    async def snapshot(self) -> tuple[float, int, int]:
        async with self._get_lock():
            return self.total_cost_usd, self.prompt_tokens, self.completion_tokens

    async def get_current_totals(self) -> tuple[float, int]:
        async with self._get_lock():
            total_tokens = self.prompt_tokens + self.completion_tokens
            return self.total_cost_usd, total_tokens
