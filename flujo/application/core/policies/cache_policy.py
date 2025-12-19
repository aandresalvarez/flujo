from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Protocol, Type

from flujo.domain.models import (
    BaseModel,
    Failure,
    Paused,
    PipelineResult,
    Quota,
    StepOutcome,
    StepResult,
    Success,
    UsageLimits,
)
from flujo.domain.outcomes import to_outcome
from flujo.infra import telemetry
from flujo.domain.dsl.cache_step import CacheStep, _generate_cache_key

from ..context_adapter import _build_context_update, _inject_context
from ..policy_registry import StepPolicy
from ..types import ExecutionFrame


class CacheStepExecutor(Protocol):
    async def execute(
        self, core: object, frame: ExecutionFrame[BaseModel]
    ) -> StepOutcome[StepResult]: ...


class DefaultCacheStepExecutor(StepPolicy[CacheStep[object, object]]):
    @property
    def handles_type(self) -> Type[CacheStep[object, object]]:
        return CacheStep

    async def execute(
        self, core: object, frame: ExecutionFrame[BaseModel]
    ) -> StepOutcome[StepResult]:
        cache_step_obj = frame.step
        if not isinstance(cache_step_obj, CacheStep):
            raise TypeError("DefaultCacheStepExecutor expects a CacheStep")
        cache_step: CacheStep[object, object] = cache_step_obj

        data = frame.data
        context = frame.context
        resources = frame.resources
        limits = frame.limits

        def _noop_setter(_pr: PipelineResult[BaseModel], _ctx: BaseModel | None) -> None:
            return None

        context_setter = frame.context_setter if callable(frame.context_setter) else _noop_setter

        try:
            cache_key = _generate_cache_key(cache_step.wrapped_step, data, context, resources)
        except Exception as e:
            telemetry.logfire.warning(
                f"Cache key generation failed for step '{cache_step.name}': {e}. Skipping cache."
            )
            cache_key = None

        if not cache_key:
            return await self._execute_uncached(
                core=core,
                cache_step=cache_step,
                data=data,
                context=context,
                resources=resources,
                limits=limits,
                context_setter=context_setter,
            )

        cache_locks_lock_fn = getattr(core, "_get_cache_locks_lock", None)
        cache_locks = getattr(core, "_cache_locks", None)
        if callable(cache_locks_lock_fn) and isinstance(cache_locks, dict):
            async with cache_locks_lock_fn():
                if cache_key not in cache_locks:
                    cache_locks[cache_key] = asyncio.Lock()
            async with cache_locks[cache_key]:
                return await self._execute_with_cache_lock(
                    core=core,
                    cache_step=cache_step,
                    cache_key=cache_key,
                    data=data,
                    context=context,
                    resources=resources,
                    limits=limits,
                    context_setter=context_setter,
                )

        return await self._execute_with_cache_lock(
            core=core,
            cache_step=cache_step,
            cache_key=cache_key,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            context_setter=context_setter,
        )

    async def _execute_uncached(
        self,
        *,
        core: object,
        cache_step: CacheStep[object, object],
        data: object,
        context: BaseModel | None,
        resources: object | None,
        limits: UsageLimits | None,
        context_setter: Callable[[PipelineResult[BaseModel], BaseModel | None], None],
    ) -> StepOutcome[StepResult]:
        quota: Quota | None = None
        try:
            get_quota = getattr(core, "_get_current_quota", None)
            if callable(get_quota):
                q = get_quota()
                quota = q if isinstance(q, Quota) else None
        except Exception:
            quota = None

        child_frame = ExecutionFrame(
            step=cache_step.wrapped_step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            quota=quota,
            stream=False,
            on_chunk=None,
            context_setter=context_setter,
            _fallback_depth=0,
        )
        execute_fn = getattr(core, "execute", None)
        if not callable(execute_fn):
            raise TypeError("ExecutorCore missing execute()")
        result_any = await execute_fn(child_frame)
        if isinstance(result_any, StepOutcome):
            return result_any
        if isinstance(result_any, StepResult):
            return to_outcome(result_any)
        raise TypeError(f"Unsupported cache step result type: {type(result_any).__name__}")

    async def _execute_with_cache_lock(
        self,
        *,
        core: object,
        cache_step: CacheStep[object, object],
        cache_key: str,
        data: object,
        context: BaseModel | None,
        resources: object | None,
        limits: UsageLimits | None,
        context_setter: Callable[[PipelineResult[BaseModel], BaseModel | None], None],
    ) -> StepOutcome[StepResult]:
        safe_step_name_fn = getattr(core, "_safe_step_name", None)

        def _safe_name(obj: object) -> str:
            if callable(safe_step_name_fn):
                try:
                    return str(safe_step_name_fn(obj))
                except Exception:
                    pass
            return str(getattr(obj, "name", cache_step.name))

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
                        validation_error = _inject_context(context, update_data, type(context))
                        if validation_error:
                            cached_result.success = False
                            cached_result.feedback = (
                                f"Context validation failed: {validation_error}"
                            )
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
            telemetry.logfire.error(f"Cache backend GET failed for step '{cache_step.name}': {e}")

        outcome = await self._execute_uncached(
            core=core,
            cache_step=cache_step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            context_setter=context_setter,
        )

        if isinstance(outcome, Success):
            step_result = outcome.step_result
        elif isinstance(outcome, Failure):
            step_result = outcome.step_result or StepResult(
                name=_safe_name(cache_step.wrapped_step),
                success=False,
                feedback=outcome.feedback,
            )
        elif isinstance(outcome, Paused):
            return outcome
        else:
            step_result = StepResult(
                name=_safe_name(cache_step.wrapped_step),
                success=False,
                feedback="Unsupported outcome",
            )

        try:
            if (
                not step_result.success
                and getattr(cache_step.wrapped_step, "updates_context", False)
                and context is not None
                and step_result.branch_context is None
            ):
                step_result.branch_context = context
        except Exception:
            pass

        if step_result.success:
            try:
                await cache_step.cache_backend.set(cache_key, step_result)
            except Exception as e:
                telemetry.logfire.error(
                    f"Cache backend SET failed for step '{cache_step.name}': {e}"
                )
        else:
            try:
                if (
                    getattr(cache_step.wrapped_step, "updates_context", False)
                    and context is not None
                    and getattr(step_result, "branch_context", None) is not None
                ):
                    bc = step_result.branch_context
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

        try:
            if context is not None and hasattr(context, "operation_count"):
                current_ops = int(getattr(context, "operation_count") or 0)
                if current_ops < 1:
                    context.operation_count = current_ops + 1
        except Exception:
            pass

        return to_outcome(step_result)
