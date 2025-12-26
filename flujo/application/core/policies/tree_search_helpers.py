from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

from flujo.domain.dsl.step import InvariantRule
from flujo.domain.dsl.tree_search import TreeSearchStep
from flujo.domain.evaluation import EvaluationReport, EvaluationScore
from flujo.domain.models import BaseModel, Checklist, PipelineContext, SearchState
from flujo.domain.scoring import ratio_score
from flujo.domain.validation import ValidationResult
from flujo.type_definitions.common import JSONObject
from flujo.utils.expressions import compile_expression_to_callable
from flujo.utils.hash import stable_digest


def _format_candidate(candidate: object) -> str:
    if isinstance(candidate, (dict, list)):
        try:
            return json.dumps(candidate, ensure_ascii=True, sort_keys=True)
        except Exception:
            return str(candidate)
    return str(candidate)


def _normalize_candidate_for_hash(candidate: object) -> object:
    if isinstance(candidate, BaseModel):
        try:
            return candidate.model_dump()
        except Exception:
            return str(candidate)
    dump_fn = getattr(candidate, "model_dump", None)
    if callable(dump_fn):
        try:
            return dump_fn()
        except Exception:
            return str(candidate)
    return candidate


def _candidate_state_hash(candidate: object) -> str:
    normalized = _normalize_candidate_for_hash(candidate)
    return stable_digest({"candidate": normalized})


def _extract_candidates(output: object) -> list[object]:
    if output is None:
        return []
    if isinstance(output, (list, tuple)):
        return list(output)
    if isinstance(output, dict):
        for key in ("candidates", "steps", "proposals"):
            if key in output and isinstance(output[key], (list, tuple)):
                return list(output[key])
        return [output]
    if isinstance(output, str):
        raw = output.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            return _extract_candidates(parsed)
        except Exception:
            lines: list[object] = []
            for line in raw.splitlines():
                stripped = line.strip(" -*\t")
                if stripped:
                    lines.append(stripped)
            return lines if lines else [cast(object, raw)]
    return [output]


def _validate_candidate(
    candidate: object, validator: Callable[[object], bool] | None
) -> tuple[bool, str | None]:
    if validator is None:
        if candidate is None:
            return False, "empty_candidate"
        if isinstance(candidate, str) and not candidate.strip():
            return False, "empty_candidate"
        if isinstance(candidate, (list, dict)) and len(candidate) == 0:
            return False, "empty_candidate"
        return True, None
    try:
        ok = validator(candidate)
    except Exception as exc:
        return False, f"validator_error:{exc}"
    if isinstance(ok, tuple):
        passed = bool(ok[0])
        reason = str(ok[1]) if len(ok) > 1 else None
        return passed, reason
    return bool(ok), None


def _build_discovery_prompt(objective: str) -> str:
    return (
        "Analyze the goal and extract hard invariants that must never be violated.\n"
        "Return a JSON array of invariant expressions or one rule per line.\n"
        f"Primary Objective: {objective}"
    )


def _normalize_invariant_output(output: object) -> list[str]:
    if output is None:
        return []
    invariants_attr = getattr(output, "invariants", None)
    if invariants_attr is not None:
        return _normalize_invariant_output(invariants_attr)
    if isinstance(output, (list, tuple)):
        return [str(item).strip() for item in output if str(item).strip()]
    if isinstance(output, dict):
        for key in ("invariants", "rules", "constraints"):
            if key in output:
                return _normalize_invariant_output(output.get(key))
        try:
            return [json.dumps(output, ensure_ascii=True, sort_keys=True)]
        except Exception:
            return [str(output)]
    if isinstance(output, str):
        raw = output.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            return _normalize_invariant_output(parsed)
        except Exception:
            lines = [line.strip(" -*\t") for line in raw.splitlines() if line.strip()]
            return lines or [raw]
    return [str(output)]


def _format_invariant_rule(rule: InvariantRule) -> str:
    if isinstance(rule, str):
        return rule
    name = getattr(rule, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return str(rule)


def _evaluate_invariant_rule(
    rule: InvariantRule,
    *,
    output: object,
    context: BaseModel | None,
) -> tuple[bool, str | None]:
    if isinstance(rule, str):
        try:
            expr_fn = compile_expression_to_callable(rule)
            return bool(expr_fn(output, context)), None
        except Exception as exc:
            return False, f"expression_error:{exc}"
    if not callable(rule):
        return False, "invalid_rule"
    try:
        return bool(rule(output, context)), None
    except TypeError:
        pass
    try:
        return bool(rule(context)), None
    except TypeError:
        pass
    try:
        return bool(rule(output)), None
    except TypeError:
        pass
    try:
        return bool(rule()), None
    except Exception as exc:
        return False, f"call_error:{exc}"


def _collect_invariants(
    step: TreeSearchStep[PipelineContext], state: SearchState
) -> list[InvariantRule]:
    rules: list[InvariantRule] = []
    rules.extend(step.static_invariants or [])
    if state.deduced_invariants:
        rules.extend(state.deduced_invariants)
    return rules


def _append_invariant_feedback(state: SearchState, violations: list[JSONObject]) -> None:
    if not violations:
        return
    existing = state.metadata.get("invariant_violations")
    if not isinstance(existing, list):
        existing = []
    existing.extend(violations)
    state.metadata["invariant_violations"] = existing[-5:]


def _build_invariant_violation(
    rule: InvariantRule,
    *,
    reason: str | None,
    node_id: str | None = None,
    candidate: object | None = None,
) -> JSONObject:
    rule_text = _format_invariant_rule(rule)
    diff: JSONObject = {"rule": rule_text}
    if reason:
        diff["reason"] = reason
    feedback = f"Invariant violated: {rule_text}"
    validation = ValidationResult(
        is_valid=False,
        score=0.0,
        diff=diff,
        feedback=feedback,
        validator_name="InvariantGuard",
    )
    record: JSONObject = {
        "rule": rule_text,
        "reason": reason,
        "diff": diff,
        "validation_result": validation.model_dump(exclude_none=True, mode="json"),
    }
    if node_id is not None:
        record["node_id"] = node_id
    if candidate is not None:
        record["candidate"] = _format_candidate(candidate)
    return record


def _build_prompt(
    *,
    objective: str,
    path_summary: str,
    candidate: object | None,
    purpose: str,
    k: int | None = None,
    invariant_violations: list[JSONObject] | None = None,
) -> str:
    lines = [f"Primary Objective: {objective}"]
    if path_summary:
        lines.append("")
        lines.append("Path Summary:")
        lines.append(path_summary)
    if invariant_violations:
        lines.append("")
        lines.append("Invariant Violations:")
        for violation in invariant_violations:
            rule = violation.get("rule")
            diff = violation.get("diff")
            reason = violation.get("reason")
            if rule is None:
                try:
                    payload = diff if diff is not None else violation
                    rule = json.dumps(payload, ensure_ascii=True, sort_keys=True)
                except Exception:
                    rule = str(diff if diff is not None else violation)
            if reason:
                lines.append(f"- {rule} ({reason})")
            else:
                lines.append(f"- {rule}")
    if candidate is not None:
        lines.append("")
        lines.append("Candidate:")
        lines.append(_format_candidate(candidate))
    if purpose == "proposer":
        lines.append("")
        if k is not None:
            lines.append(f"Propose {k} next steps.")
        else:
            lines.append("Propose the next steps.")
    return "\n".join(lines)


def _score_evaluation(output: object) -> tuple[float, bool, JSONObject]:
    score = 0.0
    hard_fail = False
    meta: JSONObject = {}
    if isinstance(output, ValidationResult):
        score = float(output.score)
        hard_fail = not output.is_valid
        try:
            meta["validation_result"] = output.model_dump(exclude_none=True)
        except Exception:
            meta["validation_result"] = {
                "is_valid": output.is_valid,
                "score": output.score,
                "feedback": output.feedback,
            }
    elif isinstance(output, Checklist):
        score = float(ratio_score(output))
        meta["checklist"] = output.model_dump()
    elif isinstance(output, EvaluationReport):
        score = float(output.score)
        hard_fail = bool(output.hard_fail)
        meta["evaluation_report"] = output.model_dump()
    elif isinstance(output, EvaluationScore):
        score = float(output.score)
        meta["evaluation_score"] = output.model_dump()
    elif isinstance(output, dict):
        meta = dict(output)
        for key in ("hard_fail", "hard_check_passed", "objective_passed", "passed"):
            if key in output and output[key] is False:
                hard_fail = True
                break
        for key in ("score", "rubric_score", "ratio"):
            if key in output and isinstance(output[key], (int, float)):
                score = float(output[key])
                break
    elif isinstance(output, (int, float)):
        score = float(output)
        meta["score"] = score
    else:
        meta["raw"] = str(output)
    score = max(0.0, min(1.0, score))
    return score, hard_fail, meta


def _diff_heuristic(output: object) -> float | None:
    diff_payload: object | None = None
    if isinstance(output, ValidationResult):
        diff_payload = output.diff
    elif isinstance(output, EvaluationReport):
        diff_payload = output.diff
    elif isinstance(output, dict):
        diff_payload = output.get("diff") or output.get("patch")
    if diff_payload is None:
        return None
    patch: object | None = None
    if isinstance(diff_payload, dict):
        patch = diff_payload.get("patch") or diff_payload.get("ops") or diff_payload.get("changes")
    elif isinstance(diff_payload, list):
        patch = diff_payload
    if isinstance(patch, list):
        return float(len(patch))
    return None


def _compute_heuristic_cost(
    output: object,
    *,
    score: float,
    hard_fail: bool,
    meta: JSONObject,
) -> float:
    diff_distance = _diff_heuristic(output)
    if diff_distance is not None:
        meta["heuristic_source"] = "diff"
        meta["diff_distance"] = diff_distance
        return float("inf") if hard_fail else float(diff_distance)
    meta["heuristic_source"] = "score"
    return float("inf") if hard_fail else max(0.0, 1.0 - score)
