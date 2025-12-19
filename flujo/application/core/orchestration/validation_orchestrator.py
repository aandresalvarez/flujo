"""Validation orchestration extracted from ExecutorCore agent path."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....domain.models import BaseModel, StepResult
from ....exceptions import PausedException

if TYPE_CHECKING:  # pragma: no cover
    from ..executor_core import ExecutorCore
    from ....domain.dsl.step import Step


class ValidationOrchestrator:
    """Runs validators and handles validation fallback semantics."""

    async def validate(
        self,
        *,
        core: "ExecutorCore[BaseModel]",
        step: "Step[object, object]",
        output: object,
        context: object | None,
        limits: object | None,
        data: object,
        attempt_context: object | None,
        attempt_resources: object | None,
        stream: bool,
        on_chunk: object | None,
        fallback_depth: int,
    ) -> StepResult | None:
        if not hasattr(step, "validators") or not step.validators:
            return None

        timeout_s = None
        try:
            cfg = getattr(step, "config", None)
            if cfg is not None and getattr(cfg, "timeout_s", None) is not None:
                timeout_s = float(cfg.timeout_s)
        except Exception:
            timeout_s = None

        try:
            await core.validator_invoker.validate(
                output, step, context=context, timeout_s=timeout_s
            )
            return None
        except PausedException:
            raise
        except Exception as e:
            fb_step = getattr(step, "fallback_step", None)
            if hasattr(fb_step, "_mock_name") and not hasattr(fb_step, "agent"):
                fb_step = None
            fb_msg = f"Validation failed: {core._format_feedback(str(e), 'Agent execution failed')}"
            original_output = output
            if fb_step is not None:
                fb_res = await core.execute(
                    step=fb_step,
                    data=data,
                    context=attempt_context,
                    resources=attempt_resources,
                    limits=limits,
                    stream=stream,
                    on_chunk=on_chunk,
                    _fallback_depth=fallback_depth + 1,
                )
                fb_res_sr = core._unwrap_outcome_to_step_result(
                    fb_res, core._safe_step_name(fb_step)
                )
                if fb_res_sr.metadata_ is None:
                    fb_res_sr.metadata_ = {}
                fb_res_sr.metadata_.update(
                    {
                        "validation_failure": fb_msg,
                        "validation_fallback_used": True,
                        "fallback_triggered": True,
                        "validators": getattr(step, "validators", []),
                        "original_error": fb_msg,
                    }
                )
                return fb_res_sr
            branch_ctx = context if getattr(step, "updates_context", False) else None
            return StepResult(
                name=core._safe_step_name(step),
                output=original_output,
                success=False,
                attempts=1,
                latency_s=0.0,
                token_counts=0,
                cost_usd=0.0,
                feedback=fb_msg,
                branch_context=branch_ctx,
                metadata_={
                    "validation_failure": fb_msg,
                    "validation_fallback_used": False,
                    "fallback_triggered": False,
                    "validators": getattr(step, "validators", []),
                    "original_error": fb_msg,
                },
                step_history=[],
            )
