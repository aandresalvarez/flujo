"""Sequential pipeline orchestration extracted from ExecutorCore."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from ....domain.models import (
    BaseModel as DomainBaseModel,
    Failure,
    Paused,
    PipelineResult,
    Quota,
    StepResult,
    Success,
    StepOutcome,
    UsageLimits,
)
from ....exceptions import PausedException, UsageLimitExceededError
from ....infra import telemetry
from ..step_history_tracker import StepHistoryTracker
from ..execution.executor_helpers import make_execution_frame


class _ExecutorCoreLike(Protocol):
    def _get_current_quota(self) -> Quota | None: ...
    def _safe_step_name(self, step: object) -> str: ...

    def execute(
        self,
        frame_or_step: object | None = None,
        data: object | None = None,
        **kwargs: object,
    ) -> Awaitable[StepOutcome[StepResult] | StepResult]: ...


class PipelineOrchestrator:
    """Runs pipelines via policy dispatch and aggregates history/context."""

    async def execute(
        self,
        *,
        core: _ExecutorCoreLike,
        pipeline: object,
        data: object,
        context: DomainBaseModel | None,
        resources: object | None,
        limits: UsageLimits | None,
        context_setter: Callable[[PipelineResult[DomainBaseModel], DomainBaseModel | None], None]
        | None,
    ) -> PipelineResult[DomainBaseModel]:
        current_data = data
        current_context = context
        history_tracker = StepHistoryTracker()

        telemetry.logfire.info(
            f"[PipelineOrchestrator] starting with {len(getattr(pipeline, 'steps', []))} steps"
        )
        for step in getattr(pipeline, "steps", []):
            try:
                telemetry.logfire.info(
                    f"[PipelineOrchestrator] executing step {getattr(step, 'name', 'unnamed')}"
                )
                frame = make_execution_frame(
                    core,
                    step,
                    current_data,
                    current_context,
                    resources,
                    limits,
                    context_setter,
                    stream=False,
                    on_chunk=None,
                    fallback_depth=0,
                    quota=core._get_current_quota(),
                    result=None,
                )
                outcome = await core.execute(frame)

                if isinstance(outcome, Success):
                    step_result = outcome.step_result
                    if not isinstance(step_result, StepResult) or getattr(
                        step_result, "name", None
                    ) in (None, "<unknown>", ""):
                        step_result = StepResult(
                            name=core._safe_step_name(step),
                            output=None,
                            success=False,
                            feedback="Missing step_result",
                        )
                elif isinstance(outcome, Failure):
                    step_result = (
                        outcome.step_result
                        if outcome.step_result is not None
                        else StepResult(
                            name=getattr(step, "name", "unknown"),
                            output=None,
                            success=False,
                            feedback=outcome.feedback or str(outcome.error),
                        )
                    )
                elif isinstance(outcome, StepResult):
                    step_result = outcome
                    if getattr(step_result, "name", None) in (None, "<unknown>", ""):
                        step_result.name = core._safe_step_name(step)
                elif isinstance(outcome, Paused):
                    raise PausedException(outcome.message)
                else:
                    step_result = StepResult(
                        name=getattr(step, "name", "unknown"),
                        output=None,
                        success=False,
                        feedback=f"Unsupported outcome type: {type(outcome).__name__}",
                    )

                history_tracker.add_step_result(step_result)

                current_data = (
                    step_result.output if step_result.output is not None else current_data
                )
                branch_ctx = getattr(step_result, "branch_context", None)
                if isinstance(branch_ctx, DomainBaseModel):
                    current_context = branch_ctx
                validation_error_obj: object | None = None
                update_mgr = getattr(core, "_context_update_manager", None)
                apply_updates = getattr(update_mgr, "apply_updates", None)
                if callable(apply_updates):
                    validation_error_obj = apply_updates(
                        step=step, output=step_result.output, context=current_context
                    )
                validation_error = (
                    validation_error_obj if isinstance(validation_error_obj, str) else None
                )
                if validation_error:
                    step_result.success = False
                    step_result.feedback = f"Context validation failed: {validation_error}"
                if not getattr(step_result, "feedback", None) and step_result.success is False:
                    step_result.feedback = "Context validation failed"

            except PausedException:
                raise
            except UsageLimitExceededError:
                raise
            except Exception as e:
                telemetry.logfire.error(
                    f"[PipelineOrchestrator] step failed: {str(e)}",
                )
                fb_text = str(e) or "Context validation failed"
                failure_result = StepResult(
                    name=getattr(step, "name", "unknown"),
                    output=None,
                    success=False,
                    attempts=1,
                    latency_s=0.0,
                    token_counts=0,
                    cost_usd=0.0,
                    feedback=fb_text,
                    branch_context=current_context,
                    metadata_={},
                )
                history_tracker.add_step_result(failure_result)
                break

        # Best-effort: ensure YAML fields persist on the final context if produced by any step
        try:
            from pydantic import BaseModel as _BM

            if isinstance(current_context, _BM):
                yt = getattr(current_context, "yaml_text", None)
                gy = getattr(current_context, "generated_yaml", None)
                if (
                    not isinstance(yt, str)
                    or not yt.strip()
                    or not isinstance(gy, str)
                    or not gy.strip()
                ):
                    for sr in reversed(history_tracker.get_history()):
                        out = getattr(sr, "output", None)
                        if isinstance(out, dict):
                            cand = out.get("yaml_text") or out.get("generated_yaml")
                            if isinstance(cand, str) and cand.strip():
                                try:
                                    if (not isinstance(yt, str) or not yt.strip()) and hasattr(
                                        current_context, "yaml_text"
                                    ):
                                        setattr(current_context, "yaml_text", cand)
                                    if (not isinstance(gy, str) or not gy.strip()) and hasattr(
                                        current_context, "generated_yaml"
                                    ):
                                        setattr(current_context, "generated_yaml", cand)
                                except Exception:
                                    pass
                                break
        except Exception:
            pass

        history = history_tracker.get_history()
        total_cost = sum(getattr(sr, "cost_usd", 0.0) or 0.0 for sr in history)
        total_tokens = sum(
            int(getattr(sr, "token_counts", 0) or 0) if hasattr(sr, "token_counts") else 0
            for sr in history
        )
        try:
            if history:
                last_branch_ctx = getattr(history[-1], "branch_context", None)
                if isinstance(last_branch_ctx, DomainBaseModel):
                    current_context = last_branch_ctx
        except Exception:
            pass

        result: PipelineResult[DomainBaseModel] = PipelineResult(
            final_output=current_data,
            final_pipeline_context=current_context,
            step_history=history,
            success=all(sr.success for sr in history),
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
        )
        try:
            if history:
                last_branch_ctx = getattr(history[-1], "branch_context", None)
                if isinstance(last_branch_ctx, DomainBaseModel):
                    result.final_pipeline_context = last_branch_ctx
        except Exception:
            pass
        return result
