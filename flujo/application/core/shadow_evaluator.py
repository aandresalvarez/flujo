from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any

from ...infra import telemetry


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

    def maybe_schedule(self, *, core: Any, step: Any, result: Any) -> None:
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
        payload = {
            "step_name": step_name,
            "success": getattr(result, "success", None),
            "feedback": getattr(result, "feedback", None),
            "output": getattr(result, "output", None),
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
        # Placeholder: future implementation can invoke Evaluator agent.
        # Today we log telemetry for observability without modifying user flows.
        telemetry.logfire.info(
            "[ShadowEval] judge invoked",
            extra={
                "model": self._config.judge_model,
                "sink": self._config.sink,
                "step": payload.get("step_name"),
            },
        )
