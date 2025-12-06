from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any

from ...infra import telemetry
from ...agents.wrapper import make_agent_async
from ...domain.evaluation import EvaluationScore


@dataclass
class ShadowEvalConfig:
    enabled: bool
    sample_rate: float
    timeout_s: float
    judge_model: str
    sink: str  # e.g., "telemetry"


class ShadowEvaluator:
    """Schedules shadow evaluations (LLM-as-judge) asynchronously without impacting user flow."""

    def __init__(
        self,
        *,
        config: ShadowEvalConfig,
        background_task_manager: Any,
    ) -> None:
        self._config = config
        self._bg = background_task_manager
        self._sampled: int = 0
        self._succeeded: int = 0
        self._failed: int = 0

    def maybe_schedule(
        self, *, core: Any, step: Any, result: Any, frame: Any | None = None
    ) -> None:
        cfg = self._config
        if not cfg.enabled or cfg.sample_rate <= 0.0:
            return
        try:
            if random.random() > cfg.sample_rate:
                return
        except Exception:
            return

        self._sampled += 1
        step_name = getattr(step, "name", "<unnamed>")
        run_id = None
        try:
            if frame is not None and hasattr(frame, "context"):
                ctx = getattr(frame, "context", None)
                run_id = getattr(ctx, "run_id", None)
            if run_id is None:
                ctx2 = getattr(core, "context", None)
                run_id = getattr(ctx2, "run_id", None)
        except Exception:
            run_id = None
        payload = {
            "step_name": step_name,
            "success": getattr(result, "success", None),
            "feedback": getattr(result, "feedback", None),
            "output": getattr(result, "output", None),
            "run_id": run_id,
        }

        async def _run_eval() -> None:
            try:
                await asyncio.wait_for(
                    self._run_judge(core=core, payload=payload), timeout=cfg.timeout_s
                )
                self._succeeded += 1
                telemetry.logfire.info(
                    "[ShadowEval] completed",
                    extra={
                        "step": step_name,
                        "sampled": self._sampled,
                        "succeeded": self._succeeded,
                        "failed": self._failed,
                    },
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._failed += 1
                telemetry.logfire.warning(
                    "[ShadowEval] failed",
                    extra={
                        "step": step_name,
                        "error": str(exc),
                        "sampled": self._sampled,
                        "succeeded": self._succeeded,
                        "failed": self._failed,
                    },
                )

        # Fire-and-forget via background task manager
        coro = _run_eval
        try:
            asyncio.create_task(coro(), name=f"shadow_eval_{step_name}")
        except Exception:
            try:
                # fallback to manager if available
                self._bg.add_task(asyncio.create_task(coro()))
            except Exception:
                pass

    async def _run_judge(self, *, core: Any, payload: dict[str, Any]) -> None:
        model = self._config.judge_model
        step_name = payload.get("step_name")
        run_id = payload.get("run_id")
        judge_prompt = (
            "You are a strict evaluator of step outputs.\n"
            "Provide a numeric score between 0.0 and 1.0 where 1.0 is perfect.\n"
            "Respond as JSON matching the schema: "
            '{"score": <float>, "reasoning": <string>, "criteria": {"<name>": <float>}}.\n'
            "Focus on correctness and safety; be concise."
        )

        agent = make_agent_async(
            model=model,
            system_prompt=judge_prompt,
            output_type=EvaluationScore,
            max_retries=1,
            timeout=int(self._config.timeout_s),
            auto_repair=True,
        )

        try:
            result = await agent.run(payload)
        except Exception as exc:
            telemetry.logfire.warning(
                "[ShadowEval] judge error",
                extra={"step": step_name, "error": str(exc)},
            )
            return

        score_obj = result if isinstance(result, EvaluationScore) else None
        score_value = getattr(score_obj, "score", None)
        reasoning = getattr(score_obj, "reasoning", None)
        criteria = getattr(score_obj, "criteria", None)

        if self._config.sink == "telemetry":
            telemetry.logfire.info(
                "[ShadowEval] judge score",
                extra={
                    "step": step_name,
                    "score": score_value,
                    "reasoning": reasoning,
                    "criteria": criteria if isinstance(criteria, dict) else None,
                },
            )
            return

        if self._config.sink == "database" and run_id:
            try:
                state_manager = getattr(core, "state_manager", None)
                if state_manager is None:
                    raise RuntimeError("state_manager not available")
                persist = getattr(state_manager, "persist_evaluation", None)
                if persist is None or not callable(persist):
                    raise RuntimeError("persist_evaluation not available on state_manager")
                await persist(
                    run_id=run_id,
                    step_name=step_name,
                    score=float(score_value) if score_value is not None else 0.0,
                    feedback=reasoning,
                    metadata=criteria if isinstance(criteria, dict) else None,
                )
                return
            except Exception as exc:
                telemetry.logfire.warning(
                    "[ShadowEval] database sink failed; falling back to telemetry",
                    extra={"step": step_name, "error": str(exc)},
                )
            telemetry.logfire.info(
                "[ShadowEval] judge score (fallback telemetry)",
                extra={
                    "step": step_name,
                    "score": score_value,
                    "reasoning": reasoning,
                    "criteria": criteria if isinstance(criteria, dict) else None,
                },
            )
