from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional, cast, Callable, Awaitable, TypeVar, Union

from pydantic import BaseModel

from ...exceptions import UsageLimitExceededError
from ...domain.models import UsageLimits, Quota
from ...domain.sandbox import SandboxProtocol
from .default_cache_components import OrjsonSerializer, Blake3Hasher
from .context_manager import ContextManager
from .types import ExecutionFrame
from ...steps.cache_step import CacheStep
from ...domain.dsl.loop import LoopStep
from ...domain.dsl.parallel import ParallelStep
from ...domain.dsl.conditional import ConditionalStep
from ...domain.dsl.dynamic_router import DynamicParallelRouterStep
from ...domain.dsl.step import HumanInTheLoopStep, Step
from ...exceptions import (
    InfiniteFallbackError,
    MissingAgentError,
    MockDetectionError,
    PausedException,
    PipelineAbortSignal,
    InfiniteRedirectError,
)
from ...domain.models import PipelineResult, StepResult, StepOutcome, Failure, Success
from ...infra.settings import get_settings
from .failure_builder import build_failure_outcome

Outcome = Union[StepOutcome[StepResult], StepResult]

TContext = TypeVar("TContext")


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

    step: Any
    data: Any
    context: Optional[Any] = None
    resources: Optional[Any] = None


StepExecutor = Callable[
    [Step[Any, Any], Any, Optional[Any], Optional[Any], Optional[Any]],
    Awaitable[StepResult],
]


def safe_step_name(step: Any) -> str:
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


def normalize_frame_context(frame: Any) -> None:
    """Retain context/resources/limits on the frame for policy access."""
    # Accessors kept for completeness; policies pull from frame directly
    _ = getattr(frame, "context", None)
    _ = getattr(frame, "resources", None)
    _ = getattr(frame, "limits", None)
    _ = getattr(frame, "on_chunk", None)


def enforce_typed_context(context: Any) -> Any:
    """Optionally enforce that context is a Pydantic BaseModel when configured.

    When FLUJO_ENFORCE_TYPED_CONTEXT=1 (or settings.enforce_typed_context), plain
    dict contexts are rejected to steer users toward typed models. By default,
    the check is advisory and logs a warning.
    """
    if context is None:
        return None
    if isinstance(context, BaseModel):
        return context

    try:
        settings = get_settings()
        enforce = bool(getattr(settings, "enforce_typed_context", False))
    except Exception:
        enforce = False

    if not enforce:
        try:
            from ...infra import telemetry as _telemetry

            _telemetry.logfire.warning(
                "Context is not a Pydantic BaseModel; typed contexts are recommended "
                "(set FLUJO_ENFORCE_TYPED_CONTEXT=1 to enforce)."
            )
        except Exception:
            pass
        return context

    raise TypeError("Context must be a Pydantic BaseModel when FLUJO_ENFORCE_TYPED_CONTEXT=1.")


def attach_sandbox_to_context(context: Any, sandbox: SandboxProtocol | None) -> None:
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


async def set_quota_and_hydrate(frame: Any, quota_manager: Any, hydration_manager: Any) -> None:
    """Assign quota to the execution context and hydrate managed state."""
    try:
        quota_manager.set_current_quota(getattr(frame, "quota", None))
    except Exception:
        pass
    try:
        await hydration_manager.hydrate_context(getattr(frame, "context", None))
    except Exception:
        pass


def get_current_quota(quota_manager: Any) -> Optional[Quota]:
    """Best-effort getter for the current quota using the manager first."""
    try:
        quota = cast(Optional[Quota], quota_manager.get_current_quota())
        if quota is not None:
            return quota
    except Exception:
        pass
    return None


def set_current_quota(quota_manager: Any, quota: Optional[Quota]) -> Optional[object]:
    """Best-effort setter for the current quota (returns token when available)."""
    try:
        return cast(Optional[object], quota_manager.set_current_quota(quota))
    except Exception:
        return None


def reset_current_quota(quota_manager: Any, token: Optional[object]) -> None:
    """Best-effort reset for quota context tokens."""
    try:
        if token is not None and hasattr(token, "old_value"):
            quota_manager.set_current_quota(token.old_value)
            return
    except Exception:
        pass


def hash_obj(obj: Any, serializer: OrjsonSerializer, hasher: Blake3Hasher) -> str:
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


async def maybe_launch_background(core: Any, frame: Any) -> Optional[Any]:
    """Launch background step if applicable and return outcome."""
    try:
        bg_outcome = await core._background_task_manager.maybe_launch_background_step(
            core=core, frame=frame
        )
    except Exception:
        return None
    return bg_outcome


def log_step_start(core: Any, step: Any, *, stream: bool, fallback_depth: int) -> None:
    """Delegate telemetry logging for step start."""
    core._telemetry_handler.log_step_start(step, stream=stream, fallback_depth=fallback_depth)


def log_execution_error(core: Any, step_name: str, exc: Exception) -> None:
    """Delegate telemetry logging for execution errors."""
    core._telemetry_handler.log_execution_error(step_name, exc)


async def execute_flow(
    core: Any, frame: ExecutionFrame[Any], called_with_frame: bool
) -> StepOutcome[StepResult] | StepResult:
    """Execute a frame through cache/dispatch/persist pipeline."""
    await core._set_quota_and_hydrate(frame)

    step = frame.step

    bg_outcome = await maybe_launch_background(core, frame)
    if bg_outcome is not None:
        return cast(Outcome, bg_outcome)

    core._normalize_frame_context(frame)
    stream = getattr(frame, "stream", False)
    _fallback_depth = getattr(frame, "_fallback_depth", 0)

    log_step_start(core, step, stream=stream, fallback_depth=_fallback_depth)

    cached_outcome, cache_key = await core._maybe_use_cache(
        frame, called_with_frame=called_with_frame
    )
    if cached_outcome is not None:
        return cast(Outcome, cached_outcome)

    try:
        result_outcome = cast(
            Outcome, await core._dispatch_frame(frame, called_with_frame=called_with_frame)
        )
    except asyncio.CancelledError:
        # Preserve cancellation semantics for signal handling tests and callers.
        raise
    except MissingAgentError as e:
        handled = core._handle_missing_agent_exception(e, step, called_with_frame=called_with_frame)
        if handled is not None:
            return cast(Outcome, handled)
        raise
    except (UsageLimitExceededError, MockDetectionError):
        raise
    except InfiniteFallbackError:
        # Control-flow exceptions must propagate to allow orchestrators to react.
        raise
    except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
        raise
    except Exception as exc:
        return cast(
            Outcome,
            core._handle_unexpected_exception(
                step=step, frame=frame, exc=exc, called_with_frame=called_with_frame
            ),
        )

    if isinstance(result_outcome, StepOutcome):
        return result_outcome
    result = result_outcome

    return cast(
        Outcome,
        await core._persist_and_finalize(
            step=step,
            result=result,
            cache_key=cache_key,
            called_with_frame=called_with_frame,
            frame=frame if called_with_frame else None,
        ),
    )


def build_failure(
    core: Any,
    *,
    step: Any,
    frame: ExecutionFrame[Any],
    exc: Exception,
    called_with_frame: bool,
) -> Failure[StepResult]:
    """Delegate failure outcome construction."""
    return build_failure_outcome(
        step=step,
        frame=frame,
        exc=exc,
        called_with_frame=called_with_frame,
        safe_step_name=core._safe_step_name,
    )


def handle_missing_agent_exception(
    core: Any, err: Any, step: Any, *, called_with_frame: bool
) -> StepOutcome[StepResult] | StepResult:
    """Delegate missing agent handling."""
    return cast(
        Outcome,
        core._result_handler.handle_missing_agent_exception(
            err, step, called_with_frame=called_with_frame
        ),
    )


async def persist_and_finalize(
    core: Any,
    *,
    step: Any,
    result: StepResult,
    cache_key: Optional[str],
    called_with_frame: bool,
    frame: ExecutionFrame[Any] | None = None,
) -> StepOutcome[StepResult] | StepResult:
    """Delegate cache persist/finalize."""
    return cast(
        Outcome,
        await core._result_handler.persist_and_finalize(
            step=step,
            result=result,
            cache_key=cache_key,
            called_with_frame=called_with_frame,
            frame=frame,
        ),
    )


def handle_unexpected_exception(
    core: Any,
    *,
    step: Any,
    frame: ExecutionFrame[Any],
    exc: Exception,
    called_with_frame: bool,
) -> StepOutcome[StepResult] | StepResult:
    """Delegate unexpected exception handling."""
    return cast(
        Outcome,
        core._result_handler.handle_unexpected_exception(
            step=step, frame=frame, exc=exc, called_with_frame=called_with_frame
        ),
    )


async def maybe_use_cache(
    core: Any, frame: ExecutionFrame[Any], *, called_with_frame: bool
) -> tuple[Optional[StepOutcome[StepResult] | StepResult], Optional[str]]:
    """Delegate cache hit retrieval."""
    cached = await core._result_handler.maybe_use_cache(frame, called_with_frame=called_with_frame)
    return cast(
        tuple[Optional[Outcome], Optional[str]],
        cached,
    )


async def execute_entrypoint(
    core: Any,
    frame_or_step: Any | None = None,
    data: Any | None = None,
    **kwargs: Any,
) -> StepOutcome[StepResult] | StepResult:
    """Entrypoint for ExecutorCore.execute (frame or step signature)."""
    called_with_frame = isinstance(frame_or_step, ExecutionFrame)
    if called_with_frame:
        frame = cast(ExecutionFrame[Any], frame_or_step)
    else:
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
            core._step_history_tracker.clear_history()
        except Exception:
            pass
        step_obj = frame_or_step if frame_or_step is not None else kwargs.get("step")
        if step_obj is None:
            raise ValueError("ExecutorCore.execute requires a Step or ExecutionFrame")
        payload = data if data is not None else kwargs.get("data")
        fb_depth_raw = kwargs.get("_fallback_depth", 0)
        try:
            fb_depth_norm = int(fb_depth_raw)
        except Exception:
            fb_depth_norm = 0
        frame = make_execution_frame(
            core,
            cast(Step[Any, Any], step_obj),
            payload,
            kwargs.get("context"),
            kwargs.get("resources"),
            kwargs.get("limits"),
            kwargs.get("context_setter"),
            stream=kwargs.get("stream", False),
            on_chunk=kwargs.get("on_chunk"),
            fallback_depth=fb_depth_norm,
            quota=kwargs.get("quota"),
            result=kwargs.get("result"),
        )

    return await execute_flow(core, frame, called_with_frame)


async def run_validation(
    core: Any,
    *,
    step: Step[Any, Any],
    output: Any,
    context: Optional[Any],
    limits: Optional[Any],
    data: Any,
    attempt_context: Optional[Any],
    attempt_resources: Optional[Any],
    stream: bool,
    on_chunk: Optional[Any],
    fallback_depth: int,
) -> Optional[StepOutcome[StepResult]]:
    """Centralized validation + fallback handling."""
    validation_result = cast(
        Optional[Union[StepOutcome[StepResult], StepResult]],
        await core._validation_orchestrator.validate(
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
        ),
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
    core: Any,
    step: Any,
    data: Any,
    context: Optional[Any],
    resources: Optional[Any],
    limits: Optional[UsageLimits],
    context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
    *,
    stream: bool = False,
    on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
    fallback_depth: int = 0,
    quota: Optional[Quota] = None,
    result: Optional[StepResult] = None,
) -> ExecutionFrame[Any]:
    """Create an ExecutionFrame with the current quota context."""
    context = enforce_typed_context(context)
    sandbox = getattr(core, "sandbox", None)
    attach_sandbox_to_context(context, sandbox)
    return ExecutionFrame(
        step=step,
        data=data,
        context=context,
        resources=resources,
        limits=limits,
        quota=quota if quota is not None else core._quota_manager.get_current_quota(),
        stream=stream,
        on_chunk=on_chunk,
        context_setter=context_setter or (lambda _res, _ctx: None),
        result=result,
        _fallback_depth=fallback_depth,
    )


async def execute_simple_step(
    core: Any,
    step: Any,
    data: Any,
    context: Optional[Any],
    resources: Optional[Any],
    limits: Optional[UsageLimits],
    stream: bool,
    on_chunk: Optional[Callable[[Any], Awaitable[None]]],
    cache_key: Optional[str],
    _fallback_depth: int | None = 0,
    _from_policy: bool = False,
) -> StepResult:
    """Compatibility shim for simple step execution via policies."""
    del _from_policy  # maintained for signature compatibility
    try:
        fb_depth = int(_fallback_depth) if _fallback_depth is not None else 0
    except Exception:
        fb_depth = 0

    frame = make_execution_frame(
        core,
        step,
        data,
        context,
        resources,
        limits,
        context_setter=None,
        stream=stream,
        on_chunk=on_chunk,
        fallback_depth=fb_depth,
        result=None,
        quota=None,
    )
    outcome = await core.simple_step_executor.execute(core, frame)
    return cast(
        StepResult, core._unwrap_outcome_to_step_result(outcome, core._safe_step_name(step))
    )


async def execute_step_compat(
    core: Any,
    step: Any,
    data: Any,
    context: Optional[Any] = None,
    resources: Optional[Any] = None,
    limits: Optional[UsageLimits] = None,
    stream: bool = False,
    on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
    context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
    result: Optional[StepResult] = None,
    _fallback_depth: int = 0,
    usage_limits: Optional[UsageLimits] = None,
) -> StepResult:
    """Legacy shim: delegate to core.execute and unwrap to StepResult."""
    if usage_limits is not None and limits is None:
        limits = usage_limits
    outcome = await core.execute(
        step,
        data,
        context=context,
        resources=resources,
        limits=limits,
        stream=stream,
        on_chunk=on_chunk,
        context_setter=context_setter,
        result=result,
        _fallback_depth=_fallback_depth,
    )
    return cast(
        StepResult, core._unwrap_outcome_to_step_result(outcome, core._safe_step_name(step))
    )


def isolate_context(context: Optional[Any], *, strict_context_isolation: bool) -> Optional[Any]:
    """Isolate context using ContextManager with strict toggle."""
    if context is None:
        return None
    if strict_context_isolation:
        isolated = ContextManager.isolate_strict(cast(Optional[BaseModel], context))
    else:
        isolated = ContextManager.isolate(cast(Optional[BaseModel], context))
    return cast(Optional[Any], isolated)


def merge_context_updates(
    main_context: Optional[Any],
    branch_context: Optional[Any],
    *,
    strict_context_merge: bool,
) -> Optional[Any]:
    """Merge two contexts using ContextManager with strict toggle."""
    if main_context is None and branch_context is None:
        return None
    if main_context is None:
        return branch_context
    if branch_context is None:
        return main_context

    if strict_context_merge:
        merged = ContextManager.merge_strict(
            cast(Optional[BaseModel], main_context),
            cast(Optional[BaseModel], branch_context),
        )
    else:
        merged = ContextManager.merge(
            cast(Optional[BaseModel], main_context),
            cast(Optional[BaseModel], branch_context),
        )
    return cast(Optional[Any], merged)


def accumulate_loop_context(
    current_context: Optional[Any],
    iteration_context: Optional[Any],
    *,
    strict_context_merge: bool,
) -> Optional[Any]:
    """Merge loop iteration context into current context."""
    if current_context is None:
        return iteration_context
    if iteration_context is None:
        return current_context
    return merge_context_updates(
        current_context, iteration_context, strict_context_merge=strict_context_merge
    )


def update_context_state(context: Optional[Any], state: str) -> None:
    """Annotate context scratchpad state best-effort."""
    if context is None:
        return
    try:
        if hasattr(context, "scratchpad"):
            try:
                scratchpad = getattr(context, "scratchpad")
                if isinstance(scratchpad, dict):
                    scratchpad["state"] = state
                else:
                    setattr(scratchpad, "state", state)
            except Exception:
                pass
    except Exception:
        pass


def is_complex_step(step: Any) -> bool:
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

    async def guard(self, limits: UsageLimits) -> None:
        async with self._get_lock():
            if (
                limits.total_cost_usd_limit is not None
                and self.total_cost_usd > limits.total_cost_usd_limit
            ):
                raise UsageLimitExceededError("Cost limit exceeded")
            total_tokens = self.prompt_tokens + self.completion_tokens
            if limits.total_tokens_limit is not None and total_tokens > limits.total_tokens_limit:
                raise UsageLimitExceededError("Token limit exceeded")

    async def snapshot(self) -> tuple[float, int, int]:
        async with self._get_lock():
            return self.total_cost_usd, self.prompt_tokens, self.completion_tokens

    async def get_current_totals(self) -> tuple[float, int]:
        async with self._get_lock():
            total_tokens = self.prompt_tokens + self.completion_tokens
            return self.total_cost_usd, total_tokens
