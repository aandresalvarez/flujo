from __future__ import annotations

from ._shared import (  # noqa: F401
    Awaitable,
    Callable,
    Dict,
    InfiniteFallbackError,
    InfiniteRedirectError,
    MissingAgentError,
    NonRetryableError,
    Optional,
    Paused,
    PausedException,
    Protocol,
    Quota,
    StepOutcome,
    StepResult,
    UsageEstimate,
    UsageLimitExceededError,
    UsageLimits,
    extract_usage_metrics,
    telemetry,
    time_perf_ns,
    time_perf_ns_to_seconds,
    to_outcome,
    _load_template_config,
    _normalize_plugin_feedback,
)
from ..policy_registry import StepPolicy
from ..types import ExecutionFrame
from ....domain.dsl.step import Step
from ....domain.models import BaseModel as DomainBaseModel


from .agent_policy_execution import prepare_agent_execution
from .agent_policy_run import run_agent_execution


# --- Agent Step Executor policy ---
class AgentStepExecutor(Protocol):
    async def execute(
        self, core: object, frame: ExecutionFrame[DomainBaseModel]
    ) -> StepOutcome[StepResult]: ...


class DefaultAgentStepExecutor(StepPolicy[Step[object, object]]):
    @property
    def handles_type(self) -> type[Step[object, object]]:
        return Step

    async def execute(
        self,
        core: object,
        frame: ExecutionFrame[DomainBaseModel],
    ) -> StepOutcome[StepResult]:
        (
            step,
            data,
            context,
            resources,
            limits,
            stream,
            on_chunk,
            cache_key,
            _fallback_depth,
        ) = await prepare_agent_execution(core, frame)
        return await run_agent_execution(
            executor=self,
            core=core,
            step=step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            stream=stream,
            on_chunk=on_chunk,
            cache_key=cache_key,
            _fallback_depth=_fallback_depth,
        )

    def _estimate_usage(
        self,
        step: object,
        data: object,
        context: object | None,
        *,
        core: object | None = None,
    ) -> UsageEstimate:
        # Prefer centralized estimator selection when available (keeps estimation logic out of policies).
        if core is not None:
            try:
                factory = getattr(core, "_estimator_factory", None)
                select_fn = getattr(factory, "select", None)
                if callable(select_fn):
                    est = select_fn(step)
                    est_fn = getattr(est, "estimate", None)
                    if callable(est_fn):
                        res = est_fn(step, data, context)
                        if isinstance(res, UsageEstimate):
                            return res
            except Exception:
                pass
            try:
                est = getattr(core, "_usage_estimator", None)
                est_fn = getattr(est, "estimate", None)
                if callable(est_fn):
                    res = est_fn(step, data, context)
                    if isinstance(res, UsageEstimate):
                        return res
            except Exception:
                pass

        # Fallback: explicit config hints
        try:
            cfg = getattr(step, "config", None)
            if cfg is not None:
                c = getattr(cfg, "expected_cost_usd", None)
                t = getattr(cfg, "expected_tokens", None)
                if c is not None or t is not None:
                    return UsageEstimate(
                        cost_usd=float(c) if c is not None else 0.0,
                        tokens=int(t) if t is not None else 0,
                    )
        except Exception:
            pass

        # Default to minimal estimate to allow execution; precise enforcement happens post-step.
        return UsageEstimate(cost_usd=0.0, tokens=0)


## Note: keep the full DefaultAgentStepExecutor.execute implementation active.
## This file previously included an experimental delegator helper which is now removed.


# --- Agent Step Executor outcomes adapter (safe, non-breaking) ---
class AgentStepExecutorOutcomes(Protocol):
    async def execute(
        self,
        core: object,
        step: object,
        data: object,
        context: object | None,
        resources: object | None,
        limits: UsageLimits | None,
        stream: bool,
        on_chunk: Callable[[object], Awaitable[None]] | None,
        cache_key: str | None,
        _fallback_depth: int = 0,
    ) -> StepOutcome[StepResult]: ...


## Legacy adapter removed: DefaultAgentStepExecutorOutcomes (native outcomes supported)
