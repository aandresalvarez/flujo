from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

from ....infra import telemetry
from ....agents.wrapper import make_agent_async
from ....domain.evaluation import EvaluationScore


@dataclass
class ShadowEvalConfig:
    enabled: bool
    sample_rate: float
    timeout_s: float
    judge_model: str
    sink: str  # e.g., "telemetry"
    evaluate_on_failure: bool = False
    run_level_enabled: bool = False


class ShadowEvaluator:
    """Schedules shadow evaluations (LLM-as-judge) asynchronously without impacting user flow."""

    def __init__(
        self,
        *,
        config: ShadowEvalConfig,
        background_task_manager: object | None,
    ) -> None:
        self._config = config
        self._bg = background_task_manager
        self._sampled: int = 0
        self._succeeded: int = 0
        self._failed: int = 0
        self._run_sampling: dict[str, bool] = {}
        self._run_finalized: dict[str, bool] = {}

    def _run_sampled(self, run_id: str | None) -> bool:
        cfg = self._config
        if run_id is None or run_id == "":
            try:
                return random.random() <= cfg.sample_rate
            except Exception:
                return False
        cached = self._run_sampling.get(run_id)
        if cached is not None:
            return cached
        try:
            sampled = random.random() <= cfg.sample_rate
        except Exception:
            sampled = False
        # Prevent unbounded growth in long-lived processes.
        if len(self._run_sampling) > 2048:
            try:
                self._run_sampling.clear()
            except Exception:
                pass
        self._run_sampling[run_id] = sampled
        return sampled

    def _mark_run_finalized(self, run_id: str | None) -> bool:
        if run_id is None or run_id == "":
            return True
        cached = self._run_finalized.get(run_id)
        if cached is not None:
            return False
        # Prevent unbounded growth in long-lived processes.
        if len(self._run_finalized) > 2048:
            try:
                self._run_finalized.clear()
            except Exception:
                pass
        self._run_finalized[run_id] = True
        return True

    def maybe_schedule(
        self,
        *,
        core: object,
        step: object,
        result: object,
        frame: object | None = None,
    ) -> None:
        cfg = self._config
        if not cfg.enabled or cfg.sample_rate <= 0.0:
            return
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

        if cfg.evaluate_on_failure:
            try:
                if bool(getattr(result, "success", False)):
                    return
            except Exception:
                return

        if not self._run_sampled(run_id):
            return

        self._sampled += 1
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

        # Fire-and-forget via background task manager; keep a handle to avoid GC.
        try:
            task = asyncio.create_task(_run_eval(), name=f"shadow_eval_{step_name}")
            if self._bg is not None:
                add_task = getattr(self._bg, "add_task", None)
                if callable(add_task):
                    add_task(task)
        except Exception:
            # Shadow eval is best-effort; swallow failures quietly.
            pass

    def maybe_schedule_run(
        self,
        *,
        core: object,
        result: object,
        run_id: str | None = None,
    ) -> None:
        cfg = self._config
        if not cfg.enabled or not cfg.run_level_enabled or cfg.sample_rate <= 0.0:
            return

        resolved_run_id = run_id
        if resolved_run_id is None or resolved_run_id == "":
            try:
                ctx = getattr(result, "final_pipeline_context", None)
                resolved_run_id = getattr(ctx, "run_id", None)
            except Exception:
                resolved_run_id = None
        if resolved_run_id is None or resolved_run_id == "":
            try:
                ctx2 = getattr(core, "context", None)
                resolved_run_id = getattr(ctx2, "run_id", None)
            except Exception:
                resolved_run_id = None

        if cfg.evaluate_on_failure:
            try:
                if bool(getattr(result, "success", False)):
                    return
            except Exception:
                return

        if not self._run_sampled(resolved_run_id):
            return
        if not self._mark_run_finalized(resolved_run_id):
            return

        self._sampled += 1

        step_summaries: list[dict[str, object]] = []
        try:
            history = getattr(result, "step_history", None)
            if isinstance(history, list):
                for sr in history[-100:]:
                    try:
                        step_summaries.append(
                            {
                                "name": getattr(sr, "name", None),
                                "success": getattr(sr, "success", None),
                                "feedback": getattr(sr, "feedback", None),
                            }
                        )
                    except Exception:
                        continue
        except Exception:
            pass

        payload = {
            "step_name": "__run__",
            "success": getattr(result, "success", None),
            "status": getattr(result, "status", None),
            "output": getattr(result, "output", None),
            "total_cost_usd": getattr(result, "total_cost_usd", None),
            "run_id": resolved_run_id,
            "step_history": step_summaries,
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
                        "step": "__run__",
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
                        "step": "__run__",
                        "error": str(exc),
                        "sampled": self._sampled,
                        "succeeded": self._succeeded,
                        "failed": self._failed,
                    },
                )

        try:
            task = asyncio.create_task(_run_eval(), name="shadow_eval___run__")
            if self._bg is not None:
                add_task = getattr(self._bg, "add_task", None)
                if callable(add_task):
                    add_task(task)
        except Exception:
            pass

    async def _run_judge(self, *, core: object, payload: dict[str, object]) -> None:
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
