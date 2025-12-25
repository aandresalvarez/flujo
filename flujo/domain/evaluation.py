from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

import inspect
import logging

from pydantic import Field

from .models import Checklist, ChecklistItem
from .scoring import ratio_score
from .validation import ValidationResult, Validator
from ..type_definitions.common import JSONObject

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class EvaluationScore(BaseModel):
    """Structured evaluation returned by the shadow judge."""

    score: float
    reasoning: Optional[str] = None
    criteria: dict[str, float] | None = None


class EvaluationReport(EvaluationScore):
    """Detailed evaluation report for multi-signal scoring."""

    rubric: Checklist | None = None
    rubric_score: float = 0.0
    objective_results: list[ValidationResult] = Field(default_factory=list)
    objective_score: float = 0.0
    hard_fail: bool = False
    diff: JSONObject | None = None
    metadata: JSONObject = Field(default_factory=dict)


class _AgentLike(Protocol):
    async def run(self, data: object | None = None, **kwargs: Any) -> object: ...


def _coerce_checklist(value: object) -> Checklist | None:
    if value is None:
        return None
    if isinstance(value, Checklist):
        return value
    if isinstance(value, dict) and "items" in value:
        try:
            return Checklist.model_validate(value)
        except Exception:
            return None
    return None


def _ensure_checklist_items(checklist: Checklist) -> Checklist:
    items = list(checklist.items)
    cleaned: list[ChecklistItem] = []
    for item in items:
        if isinstance(item, ChecklistItem):
            cleaned.append(item)
        elif isinstance(item, dict):
            try:
                cleaned.append(ChecklistItem.model_validate(item))
            except Exception as exc:
                logger.warning(
                    "Checklist item validation failed",
                    extra={"item": item, "error": str(exc)},
                )
    return checklist.model_copy(update={"items": cleaned})


async def _invoke_agent(agent: _AgentLike | Callable[..., object], payload: object) -> object:
    if hasattr(agent, "run") and callable(getattr(agent, "run")):
        return await getattr(agent, "run")(payload)
    if callable(agent):
        result = agent(payload)
        if inspect.isawaitable(result):
            return await result
        return result
    raise TypeError("Agent does not expose a callable or run() method")


class MultiSignalEvaluator:
    """Evaluate solutions with rubric and objective signals."""

    name: str = "MultiSignalEvaluator"

    def __init__(
        self,
        *,
        review_agent: _AgentLike | Callable[..., object] | None = None,
        validator_agent: _AgentLike | Callable[..., object] | None = None,
        objective_validators: list[Validator] | None = None,
        rubric_weight: float = 1.0,
        objective_weight: float = 2.0,
        strict_objective: bool = True,
        diff_processor: object | None = None,
    ) -> None:
        self.review_agent = review_agent
        self.validator_agent = validator_agent
        self.objective_validators = list(objective_validators or [])
        self.rubric_weight = float(rubric_weight)
        self.objective_weight = float(objective_weight)
        self.strict_objective = bool(strict_objective)
        self.diff_processor = diff_processor

    async def run(self, data: object | None = None, **kwargs: Any) -> EvaluationReport:
        context = kwargs.get("context")
        payload: dict[str, object] = data if isinstance(data, dict) else {"solution": data}

        solution = payload.get("solution")
        if solution is None:
            solution = payload.get("output", payload.get("candidate", data))

        task = payload.get("task") or payload.get("prompt")
        if task is None and context is not None:
            task = getattr(context, "initial_prompt", None)

        checklist = _coerce_checklist(payload.get("checklist"))
        if checklist is None and self.review_agent is not None:
            review_input = task if task is not None else solution
            try:
                review_raw = await _invoke_agent(self.review_agent, review_input)
                checklist = _coerce_checklist(review_raw)
            except Exception:
                checklist = None

        if checklist is None:
            checklist = Checklist(items=[])
        checklist = _ensure_checklist_items(checklist)

        if self.validator_agent is not None:
            try:
                validated_raw = await _invoke_agent(
                    self.validator_agent,
                    {"solution": solution, "checklist": checklist},
                )
                validated = _coerce_checklist(validated_raw)
                if validated is not None:
                    checklist = _ensure_checklist_items(validated)
            except Exception:
                pass

        rubric_score = float(ratio_score(checklist))

        validators = payload.get("objective_validators") or payload.get("validators")
        if validators is None:
            validators = self.objective_validators
        if not isinstance(validators, list):
            validators = []

        objective_results: list[ValidationResult] = []
        for validator in validators:
            try:
                validate_fn = getattr(validator, "validate", None) or validator
                res = validate_fn(solution, context=context)
                if inspect.isawaitable(res):
                    res = await res
                if isinstance(res, ValidationResult):
                    objective_results.append(res)
                elif hasattr(res, "is_valid"):
                    result_payload: dict[str, object] = {
                        "is_valid": bool(getattr(res, "is_valid")),
                        "feedback": getattr(res, "feedback", None),
                        "validator_name": getattr(validator, "name", type(validator).__name__),
                    }
                    score = getattr(res, "score", None)
                    if score is not None:
                        try:
                            result_payload["score"] = float(score)
                        except Exception:
                            pass
                    diff = getattr(res, "diff", None)
                    if diff is not None:
                        result_payload["diff"] = diff
                    objective_results.append(ValidationResult(**result_payload))
                else:
                    objective_results.append(
                        ValidationResult(
                            is_valid=bool(res),
                            feedback=None,
                            validator_name=getattr(validator, "name", type(validator).__name__),
                        )
                    )
            except Exception as exc:
                objective_results.append(
                    ValidationResult(
                        is_valid=False,
                        feedback=f"Validator {type(validator).__name__} failed: {exc}",
                        validator_name=getattr(validator, "name", type(validator).__name__),
                    )
                )

        objective_score = 0.0
        if objective_results:
            passed = sum(1 for res in objective_results if res.is_valid)
            objective_score = passed / len(objective_results)

        hard_fail = self.strict_objective and any(not res.is_valid for res in objective_results)

        total_weight = 0.0
        score_sum = 0.0
        if self.rubric_weight > 0:
            total_weight += self.rubric_weight
            score_sum += rubric_score * self.rubric_weight
        if objective_results and self.objective_weight > 0:
            total_weight += self.objective_weight
            score_sum += objective_score * self.objective_weight

        score = score_sum / total_weight if total_weight else 0.0
        if hard_fail:
            score = 0.0

        diff_payload: JSONObject | None = None
        expected = payload.get("expected") or payload.get("target")
        if expected is not None and self.diff_processor is not None:
            try:
                process_fn = getattr(self.diff_processor, "process", None)
                if callable(process_fn):
                    diff_result = process_fn(
                        {"before": solution, "after": expected}, context=context
                    )
                elif callable(self.diff_processor):
                    diff_result = self.diff_processor(
                        {"before": solution, "after": expected}, context=context
                    )
                else:
                    diff_result = None
                if inspect.isawaitable(diff_result):
                    diff_result = await diff_result
                if isinstance(diff_result, dict):
                    diff_payload = diff_result
            except Exception:
                diff_payload = None

        return EvaluationReport(
            score=float(score),
            rubric=checklist,
            rubric_score=rubric_score,
            objective_results=objective_results,
            objective_score=objective_score,
            hard_fail=hard_fail,
            diff=diff_payload,
            criteria={"rubric": rubric_score, "objective": objective_score},
        )


def make_multi_signal_evaluator(
    *,
    review_agent: _AgentLike | Callable[..., object] | None = None,
    validator_agent: _AgentLike | Callable[..., object] | None = None,
    objective_validators: list[Validator] | None = None,
    rubric_weight: float = 1.0,
    objective_weight: float = 2.0,
    strict_objective: bool = True,
    diff_processor: object | None = None,
) -> MultiSignalEvaluator:
    if review_agent is None or validator_agent is None:
        try:
            from ..agents import make_review_agent, make_validator_agent
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError("Unable to resolve default review/validator agents") from exc
        review_agent = review_agent or make_review_agent()
        validator_agent = validator_agent or make_validator_agent()
    return MultiSignalEvaluator(
        review_agent=review_agent,
        validator_agent=validator_agent,
        objective_validators=objective_validators,
        rubric_weight=rubric_weight,
        objective_weight=objective_weight,
        strict_objective=strict_objective,
        diff_processor=diff_processor,
    )
