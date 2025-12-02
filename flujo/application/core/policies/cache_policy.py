from __future__ import annotations
# mypy: ignore-errors

from typing import Type

from ._shared import (  # noqa: F401
    Any,
    Awaitable,
    Callable,
    Failure,
    Paused,
    Optional,
    PipelineResult,
    Protocol,
    StepOutcome,
    StepResult,
    UsageLimits,
    Success,
    CacheStep,
    _build_context_update,
    _generate_cache_key,
    _inject_context,
    asyncio,
    telemetry,
    to_outcome,
)
from ..policy_registry import StepPolicy
from ..types import ExecutionFrame

# --- Cache Step Executor policy ---


class CacheStepExecutor(Protocol):
    async def execute(
        self,
        core: Any,
        cache_step: CacheStep[Any, Any],
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        step_executor: Optional[Callable[..., Awaitable[StepResult]]],
    ) -> StepOutcome[StepResult]: ...


class DefaultCacheStepExecutor(StepPolicy[CacheStep]):
    @property
    def handles_type(self) -> Type[CacheStep]:
        return CacheStep

    async def execute(
        self,
        core: Any,
        cache_step: CacheStep[Any, Any],
        data: Any | None = None,
        context: Optional[Any] = None,
        resources: Optional[Any] = None,
        limits: Optional[UsageLimits] = None,
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
        step_executor: Optional[Callable[..., Awaitable[StepResult]]] = None,
        # Backward-compat: retain 'step' parameter for legacy inspection tooling
        step: Optional[Any] = None,
    ) -> StepOutcome[StepResult]:
        """Handle CacheStep execution with concurrency control and resilience."""
        if isinstance(cache_step, ExecutionFrame):
            frame = cache_step
            cache_step = frame.step  # type: ignore[assignment]
            data = frame.data
            context = frame.context
            resources = frame.resources
            limits = frame.limits
            context_setter = getattr(frame, "context_setter", None)

        try:
            cache_key = _generate_cache_key(cache_step.wrapped_step, data, context, resources)
        except Exception as e:
            telemetry.logfire.warning(
                f"Cache key generation failed for step '{cache_step.name}': {e}. Skipping cache."
            )
            cache_key = None
        if cache_key:
            async with core._cache_locks_lock:
                if cache_key not in core._cache_locks:
                    core._cache_locks[cache_key] = asyncio.Lock()
            async with core._cache_locks[cache_key]:
                try:
                    cached_result = await cache_step.cache_backend.get(cache_key)
                    if cached_result is not None:
                        if cached_result.metadata_ is None:
                            cached_result.metadata_ = {"cache_hit": True}
                        else:
                            cached_result.metadata_["cache_hit"] = True
                        if cached_result.branch_context is not None and context is not None:
                            update_data = _build_context_update(cached_result.output)
                            if update_data:
                                validation_error = _inject_context(
                                    context, update_data, type(context)
                                )
                                if validation_error:
                                    cached_result.success = False
                                    cached_result.feedback = (
                                        f"Context validation failed: {validation_error}"
                                    )
                            # Honor optional sink_to on wrapped simple steps when resuming from cache
                            try:
                                sink_path = getattr(cache_step.wrapped_step, "sink_to", None)
                                if sink_path:
                                    from flujo.utils.context import (
                                        set_nested_context_field as _set_field,
                                    )

                                    try:
                                        _set_field(context, str(sink_path), cached_result.output)
                                    except Exception:
                                        if "." not in str(sink_path):
                                            object.__setattr__(
                                                context, str(sink_path), cached_result.output
                                            )
                            except Exception:
                                pass
                        return to_outcome(cached_result)
                except Exception as e:
                    telemetry.logfire.error(
                        f"Cache backend GET failed for step '{cache_step.name}': {e}"
                    )
                quota = None
                try:
                    if hasattr(core, "_get_current_quota"):
                        quota = core._get_current_quota()
                except Exception:
                    quota = None

                frame = ExecutionFrame(
                    step=cache_step.wrapped_step,
                    data=data,
                    context=context,
                    resources=resources,
                    limits=limits,
                    quota=quota,
                    stream=False,
                    on_chunk=None,
                    context_setter=(
                        context_setter if context_setter is not None else (lambda _pr, _ctx: None)
                    ),
                    _fallback_depth=0,
                )
                result = await core.execute(frame)
                # Normalize to StepResult if policy returned typed outcome
                if isinstance(result, StepOutcome):
                    if isinstance(result, Success):
                        result = result.step_result
                    elif isinstance(result, Failure):
                        result = result.step_result or StepResult(
                            name=core._safe_step_name(cache_step.wrapped_step),
                            success=False,
                            feedback=result.feedback,
                        )
                    elif isinstance(result, Paused):
                        return result
                    else:
                        result = StepResult(
                            name=core._safe_step_name(cache_step.wrapped_step),
                            success=False,
                            feedback="Unsupported outcome",
                        )
                # Preserve per-attempt context mutations for updates_context even on failure
                try:
                    if (
                        not result.success
                        and getattr(cache_step.wrapped_step, "updates_context", False)
                        and context is not None
                        and result.branch_context is None
                    ):
                        result.branch_context = context
                except Exception:
                    pass
                if result.success:
                    try:
                        await cache_step.cache_backend.set(cache_key, result)
                    except Exception as e:
                        telemetry.logfire.error(
                            f"Cache backend SET failed for step '{cache_step.name}': {e}"
                        )
                else:
                    # Failure path: proactively reflect branch_context mutations onto the
                    # live context for updates_context semantics (e.g., increment counters)
                    try:
                        if (
                            getattr(cache_step.wrapped_step, "updates_context", False)
                            and context is not None
                            and getattr(result, "branch_context", None) is not None
                        ):
                            bc = result.branch_context
                            cm = type(context)
                            fields = getattr(cm, "model_fields", {})
                            for fname in fields.keys():
                                try:
                                    bval = getattr(bc, fname, None)
                                    if (
                                        isinstance(bval, (int, float, str, bool))
                                        and getattr(context, fname, None) != bval
                                    ):
                                        setattr(context, fname, bval)
                                except Exception:
                                    continue
                    except Exception:
                        pass
                # Increment operation count on context when present (even on failure)
                try:
                    if context is not None and hasattr(context, "operation_count"):
                        current_ops = int(getattr(context, "operation_count") or 0)
                        if current_ops < 1:
                            context.operation_count = current_ops + 1
                except Exception:
                    pass
                return to_outcome(result)
        frame = ExecutionFrame(
            step=cache_step.wrapped_step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            quota=(core._get_current_quota() if hasattr(core, "_get_current_quota") else None),
            stream=False,
            on_chunk=None,
            context_setter=(
                context_setter if context_setter is not None else (lambda _pr, _ctx: None)
            ),
            _fallback_depth=0,
        )
        # Ensure we return according to requested mode
        result = await core.execute(frame)
        if isinstance(result, StepOutcome):
            return result
        return to_outcome(result)


# --- End Cache Step Executor policy ---
