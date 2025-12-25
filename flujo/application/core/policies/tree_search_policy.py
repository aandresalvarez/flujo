from __future__ import annotations

from typing import Callable, Protocol, cast
import json
import math
import time

from flujo.application.conversation.history_manager import HistoryManager
from flujo.application.core.context.context_manager import ContextManager
from flujo.application.core.policy_registry import StepPolicy
from flujo.application.core.runtime.quota_manager import build_root_quota
from flujo.application.core.runtime.usage_messages import format_reservation_denial
from flujo.application.core.types import ExecutionFrame
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import InvariantRule, Step
from flujo.domain.dsl.tree_search import TreeSearchStep
from flujo.domain.evaluation import EvaluationReport, EvaluationScore
from flujo.domain.models import (
    BaseModel,
    Checklist,
    Failure,
    PipelineContext,
    PipelineResult,
    Quota,
    QuotaExceededError,
    SearchNode,
    SearchState,
    Success,
    StepOutcome,
    StepResult,
    UsageEstimate,
    UsageLimits,
)
from flujo.domain.validation import ValidationResult
from flujo.domain.scoring import ratio_score
from flujo.exceptions import (
    InfiniteRedirectError,
    PausedException,
    PipelineAbortSignal,
    UsageLimitExceededError,
)
from flujo.infra.settings import get_settings
from flujo.infra import telemetry
from flujo.type_definitions.common import JSONObject
from flujo.utils.context import get_excluded_fields, safe_merge_context_updates
from flujo.utils.expressions import compile_expression_to_callable
from flujo.utils.hash import stable_digest


class TreeSearchStepExecutor(Protocol):
    async def execute(
        self,
        core: object,
        frame: ExecutionFrame[BaseModel],
    ) -> StepOutcome[StepResult]: ...


def _ensure_context(context: BaseModel | None, data: object) -> PipelineContext:
    if isinstance(context, PipelineContext):
        ctx = context
    else:
        ctx = PipelineContext(initial_prompt=str(data))
    if not getattr(ctx, "initial_prompt", None):
        try:
            ctx.initial_prompt = str(data)
        except Exception:
            pass
    return ctx


def _coerce_step(obj: object, *, name: str) -> Step[object, object] | Pipeline[object, object]:
    if isinstance(obj, Step) or isinstance(obj, Pipeline):
        return obj
    if callable(obj) or hasattr(obj, "run"):
        return Step(name=name, agent=obj)
    raise ValueError(f"{name} must be a Step, Pipeline, or callable agent")


def _force_temperature(
    step: Step[object, object],
    *,
    temperature: float = 0.0,
) -> Step[object, object]:
    try:
        cfg = getattr(step, "config", None)
        if cfg is not None and getattr(cfg, "temperature", None) == temperature:
            return step
        new_cfg = cfg.model_copy() if cfg is not None else None
        if new_cfg is not None:
            new_cfg.temperature = temperature
        return step.model_copy(update={"config": new_cfg})
    except Exception:
        try:
            step.config.temperature = temperature
        except Exception:
            pass
        return step


def _estimate_usage(
    core: object,
    step: object,
    data: object,
    context: object | None,
) -> UsageEstimate:
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
    try:
        cfg = getattr(step, "config", None)
        c = getattr(cfg, "expected_cost_usd", None)
        t = getattr(cfg, "expected_tokens", None)
        if c is not None or t is not None:
            return UsageEstimate(
                cost_usd=float(c) if c is not None else 0.0,
                tokens=int(t) if t is not None else 0,
            )
    except Exception:
        pass
    return UsageEstimate(cost_usd=0.0, tokens=0)


def _reserve_quota(
    quota: Quota | None,
    estimate: UsageEstimate,
    limits: UsageLimits | None,
) -> None:
    if quota is None:
        return
    if not quota.reserve(estimate):
        rem_cost, rem_tokens = quota.get_remaining()
        denial = format_reservation_denial(
            estimate, limits, remaining=(float(rem_cost), int(rem_tokens))
        )
        raise UsageLimitExceededError(denial.human)


def _reconcile_quota(
    quota: Quota | None,
    estimate: UsageEstimate,
    actual: UsageEstimate,
    limits: UsageLimits | None,
) -> None:
    if quota is None:
        return
    try:
        quota.reclaim(estimate, actual)
    except QuotaExceededError as exc:
        denial = format_reservation_denial(
            UsageEstimate(cost_usd=exc.extra_cost_usd, tokens=exc.extra_tokens),
            limits,
            remaining=(exc.remaining_cost_usd, exc.remaining_tokens),
        )
        raise UsageLimitExceededError(denial.human) from None


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
        "validation_result": validation.model_dump(exclude_none=True),
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


def _compute_cost(
    cost_fn: Callable[..., float] | None,
    *,
    candidate: object,
    parent: SearchNode,
    depth: int,
    evaluation: object | None,
) -> float:
    if cost_fn is None:
        return float(depth)
    try:
        return float(cost_fn(candidate, parent, depth, evaluation))
    except TypeError:
        try:
            return float(cost_fn(candidate, depth))
        except TypeError:
            return float(cost_fn(candidate))


def _should_accept_goal(score: float, threshold: float, hard_fail: bool) -> bool:
    if hard_fail:
        return False
    return score >= threshold


def _get_best_node(
    current: SearchNode | None,
    challenger: SearchNode,
    score: float,
) -> SearchNode:
    if current is None:
        return challenger
    try:
        current_score = float(current.metadata.get("rubric_score", 0.0))
    except Exception:
        current_score = 0.0
    if score > current_score:
        return challenger
    if score == current_score and challenger.f_cost < current.f_cost:
        return challenger
    return current


class DefaultTreeSearchStepExecutor(StepPolicy[TreeSearchStep[PipelineContext]]):
    @property
    def handles_type(self) -> type[TreeSearchStep[PipelineContext]]:
        return TreeSearchStep

    async def execute(
        self,
        core: object,
        frame: ExecutionFrame[BaseModel],
    ) -> StepOutcome[StepResult]:
        step = frame.step
        data = frame.data
        resources = frame.resources
        limits = frame.limits

        if not isinstance(step, TreeSearchStep):
            raise ValueError(f"Expected TreeSearchStep, got {type(step)}")

        start_time = time.monotonic()
        context = _ensure_context(frame.context, data)

        proposer = _coerce_step(step.proposer, name=f"{step.name}_proposer")
        evaluator = _coerce_step(step.evaluator, name=f"{step.name}_evaluator")
        if isinstance(proposer, Step):
            proposer = _force_temperature(proposer)
        if isinstance(evaluator, Step):
            evaluator = _force_temperature(evaluator)

        # Resolve quota from current execution context
        quota: Quota | None = None
        try:
            if hasattr(core, "_get_current_quota"):
                quota = core._get_current_quota()
        except Exception:
            quota = None
        if quota is None:
            quota = getattr(frame, "quota", None)
        if quota is None and limits is not None:
            try:
                quota = build_root_quota(limits)
            except Exception:
                quota = None

        state = getattr(context, "tree_search_state", None)
        if not isinstance(state, SearchState):
            state = SearchState()
            context.tree_search_state = state

        if not state.nodes:
            root_id = f"n{state.next_node_id}"
            state.next_node_id += 1
            root = SearchNode(
                node_id=root_id,
                parent_id=None,
                depth=0,
                candidate=data,
                output=data,
                g_cost=0.0,
                h_cost=1.0,
                f_cost=1.0,
                state_hash=_candidate_state_hash(data),
            )
            root.attach_context(context)
            state.nodes[root_id] = root
            state.open_set = [root_id]
            state.closed_set = [root.state_hash]

        best_node: SearchNode | None = None
        if state.best_node_id and state.best_node_id in state.nodes:
            best_node = state.nodes[state.best_node_id]

        search_objective = str(getattr(context, "initial_prompt", "") or str(data))

        hm = HistoryManager()
        total_cost = 0.0
        total_tokens = 0
        step_history: list[StepResult] = []
        goal_reached = False
        winner: SearchNode | None = None

        def _record_trace(event: JSONObject) -> None:
            try:
                state.trace.append(event)
            except Exception:
                pass

        def _snapshot_state() -> None:
            context.tree_search_state = state

        async def _persist_frontier_state(status: str = "running") -> None:
            state_manager = getattr(core, "state_manager", None)
            persist = getattr(state_manager, "persist_workflow_state", None)
            run_id = getattr(context, "run_id", None)
            step_index = getattr(context, "current_step_index", None)
            if not callable(persist) or run_id is None or step_index is None:
                return
            try:
                await persist(
                    run_id=run_id,
                    context=context,
                    current_step_index=int(step_index),
                    last_step_output=None,
                    status=status,
                    step_history=step_history,
                )
            except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
                raise
            except Exception:
                return

        if step.discovery_agent is not None:
            settings = get_settings()
            discovery_enabled = bool(getattr(settings, "tree_search_discovery_enabled", True))
            if discovery_enabled and not bool(state.metadata.get("discovery_complete", False)):
                discovery = _coerce_step(step.discovery_agent, name=f"{step.name}_discovery")
                if isinstance(discovery, Step):
                    discovery = _force_temperature(discovery)
                discovery_prompt = _build_discovery_prompt(search_objective)
                try:
                    est_disc = _estimate_usage(core, discovery, discovery_prompt, context)
                    _reserve_quota(quota, est_disc, limits)
                    disc_out = None
                    disc_sr: StepResult
                    if isinstance(discovery, Pipeline):
                        exec_pipeline = getattr(core, "_execute_pipeline_via_policies", None)
                        if not callable(exec_pipeline):
                            raise TypeError("ExecutorCore missing _execute_pipeline_via_policies")
                        pr = await exec_pipeline(
                            discovery,
                            discovery_prompt,
                            context,
                            resources,
                            limits,
                            context_setter=None,
                        )
                        disc_out = getattr(pr, "final_output", None)
                        if disc_out is None and pr.step_history:
                            disc_out = pr.step_history[-1].output
                        disc_sr = StepResult(
                            name=getattr(discovery, "name", "discovery"),
                            output=disc_out,
                            success=pr.success,
                            attempts=len(pr.step_history),
                            latency_s=0.0,
                            token_counts=int(getattr(pr, "total_tokens", 0) or 0),
                            cost_usd=float(getattr(pr, "total_cost_usd", 0.0) or 0.0),
                            step_history=list(pr.step_history),
                        )
                    else:
                        execute_step = getattr(core, "execute", None)
                        if not callable(execute_step):
                            raise TypeError("ExecutorCore missing execute")
                        outcome = await execute_step(
                            step=discovery,
                            data=discovery_prompt,
                            context=context,
                            resources=resources,
                            limits=limits,
                            context_setter=None,
                            stream=False,
                            on_chunk=None,
                        )
                        unwrap_fn = getattr(core, "_unwrap_outcome_to_step_result", None)
                        if not callable(unwrap_fn):
                            raise TypeError("ExecutorCore missing _unwrap_outcome_to_step_result")
                        disc_sr = unwrap_fn(outcome, getattr(discovery, "name", "discovery"))
                        disc_out = disc_sr.output
                    actual_disc = UsageEstimate(
                        cost_usd=float(getattr(disc_sr, "cost_usd", 0.0) or 0.0),
                        tokens=int(getattr(disc_sr, "token_counts", 0) or 0),
                    )
                    _reconcile_quota(quota, est_disc, actual_disc, limits)
                except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
                    state.status = "paused"
                    _snapshot_state()
                    await _persist_frontier_state(status="paused")
                    raise

                step_history.append(disc_sr)
                total_cost += float(getattr(disc_sr, "cost_usd", 0.0) or 0.0)
                total_tokens += int(getattr(disc_sr, "token_counts", 0) or 0)

                deduced = _normalize_invariant_output(disc_out)
                state.deduced_invariants = deduced
                state.metadata["discovery_complete"] = True
                _record_trace(
                    {
                        "event": "invariants_discovered",
                        "count": len(deduced),
                    }
                )

        while True:
            if step.max_iterations is not None and state.iterations >= step.max_iterations:
                break
            current = state.pop_best_open()
            if current is None:
                break
            state.iterations += 1

            node_ctx = current.rehydrate_context(type(context))
            if node_ctx is None:
                isolated = ContextManager.isolate(
                    context, purpose=f"tree_search:{step.name}:{current.node_id}"
                )
                node_ctx = isolated if isinstance(isolated, PipelineContext) else None
            if node_ctx is None:
                node_ctx = context

            invariant_rules = _collect_invariants(step, state)
            if invariant_rules:
                violations: list[JSONObject] = []
                node_output = current.output if current.output is not None else current.candidate
                for rule in invariant_rules:
                    ok, reason = _evaluate_invariant_rule(
                        rule, output=node_output, context=node_ctx
                    )
                    if not ok:
                        violations.append(
                            _build_invariant_violation(
                                rule,
                                reason=reason,
                                node_id=current.node_id,
                                candidate=node_output,
                            )
                        )
                if violations:
                    current.f_cost = float("inf")
                    current.h_cost = float("inf")
                    current.metadata["invariant_violations"] = [
                        v.get("rule") for v in violations if v.get("rule") is not None
                    ]
                    _append_invariant_feedback(state, violations)
                    _record_trace(
                        {
                            "event": "invariant_violation",
                            "node_id": current.node_id,
                            "violations": violations,
                        }
                    )
                    _snapshot_state()
                    continue

            try:
                score_val = float(current.metadata.get("rubric_score", 0.0))
            except Exception:
                score_val = 0.0
            hard_fail = bool(current.metadata.get("hard_fail", False))
            if _should_accept_goal(score_val, step.goal_score_threshold, hard_fail):
                winner = current
                goal_reached = True
                break
            if current.depth >= step.max_depth:
                _record_trace(
                    {"event": "max_depth", "node_id": current.node_id, "depth": current.depth}
                )
                continue

            path_nodes: list[SearchNode] = []
            cur: SearchNode | None = current
            while cur is not None:
                path_nodes.append(cur)
                parent_id = cur.parent_id
                cur = state.nodes.get(parent_id) if parent_id is not None else None
            path_nodes.reverse()
            path_texts = [
                _format_candidate(p.output if p.output is not None else p.candidate)
                for p in path_nodes
            ]
            path_summary = hm.summarize(path_texts, max_tokens=step.path_max_tokens)
            objective = search_objective

            proposer_prompt = _build_prompt(
                objective=objective,
                path_summary=path_summary,
                candidate=current.output if current.output is not None else current.candidate,
                purpose="proposer",
                k=step.branching_factor,
                invariant_violations=(
                    state.metadata.get("invariant_violations")
                    if isinstance(state.metadata.get("invariant_violations"), list)
                    else None
                ),
            )

            try:
                est = _estimate_usage(core, proposer, proposer_prompt, node_ctx)
                _reserve_quota(quota, est, limits)
                proposer_out = None
                proposer_sr: StepResult
                if isinstance(proposer, Pipeline):
                    exec_pipeline = getattr(core, "_execute_pipeline_via_policies", None)
                    if not callable(exec_pipeline):
                        raise TypeError("ExecutorCore missing _execute_pipeline_via_policies")
                    pr = await exec_pipeline(
                        proposer,
                        proposer_prompt,
                        node_ctx,
                        resources,
                        limits,
                        context_setter=None,
                    )
                    proposer_out = getattr(pr, "final_output", None)
                    if proposer_out is None and pr.step_history:
                        proposer_out = pr.step_history[-1].output
                    proposer_sr = StepResult(
                        name=getattr(proposer, "name", "proposer"),
                        output=proposer_out,
                        success=pr.success,
                        attempts=len(pr.step_history),
                        latency_s=0.0,
                        token_counts=int(getattr(pr, "total_tokens", 0) or 0),
                        cost_usd=float(getattr(pr, "total_cost_usd", 0.0) or 0.0),
                        step_history=list(pr.step_history),
                    )
                else:
                    execute_step = getattr(core, "execute", None)
                    if not callable(execute_step):
                        raise TypeError("ExecutorCore missing execute")
                    outcome = await execute_step(
                        step=proposer,
                        data=proposer_prompt,
                        context=node_ctx,
                        resources=resources,
                        limits=limits,
                        context_setter=None,
                        stream=False,
                        on_chunk=None,
                    )
                    unwrap_fn = getattr(core, "_unwrap_outcome_to_step_result", None)
                    if not callable(unwrap_fn):
                        raise TypeError("ExecutorCore missing _unwrap_outcome_to_step_result")
                    proposer_sr = unwrap_fn(outcome, getattr(proposer, "name", "proposer"))
                    proposer_out = proposer_sr.output
                actual = UsageEstimate(
                    cost_usd=float(getattr(proposer_sr, "cost_usd", 0.0) or 0.0),
                    tokens=int(getattr(proposer_sr, "token_counts", 0) or 0),
                )
                _reconcile_quota(quota, est, actual, limits)
            except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
                state.status = "paused"
                _snapshot_state()
                await _persist_frontier_state(status="paused")
                raise

            step_history.append(proposer_sr)
            total_cost += float(getattr(proposer_sr, "cost_usd", 0.0) or 0.0)
            total_tokens += int(getattr(proposer_sr, "token_counts", 0) or 0)
            candidates = _extract_candidates(proposer_out)
            if step.branching_factor and len(candidates) > step.branching_factor:
                candidates = candidates[: step.branching_factor]

            closed_hashes = set(state.closed_set)
            seen_hashes: set[str] = set()
            expanded = 0

            def _mark_closed(state_hash: str) -> None:
                if state_hash in closed_hashes:
                    return
                closed_hashes.add(state_hash)
                state.closed_set.append(state_hash)

            for candidate in candidates:
                ok, reason = _validate_candidate(candidate, step.candidate_validator)
                if not ok:
                    _record_trace(
                        {
                            "event": "candidate_skipped",
                            "node_id": current.node_id,
                            "reason": reason or "invalid",
                        }
                    )
                    continue

                state_hash = _candidate_state_hash(candidate)
                if state_hash in closed_hashes or state_hash in seen_hashes:
                    _record_trace(
                        {
                            "event": "candidate_deduped",
                            "node_id": current.node_id,
                            "hash": state_hash,
                        }
                    )
                    continue
                seen_hashes.add(state_hash)

                if invariant_rules:
                    candidate_violations: list[JSONObject] = []
                    for rule in invariant_rules:
                        ok, reason = _evaluate_invariant_rule(
                            rule, output=candidate, context=node_ctx
                        )
                        if not ok:
                            candidate_violations.append(
                                _build_invariant_violation(
                                    rule,
                                    reason=reason,
                                    node_id=current.node_id,
                                    candidate=candidate,
                                )
                            )
                    if candidate_violations:
                        _append_invariant_feedback(state, candidate_violations)
                        _record_trace(
                            {
                                "event": "invariant_violation",
                                "node_id": current.node_id,
                                "violations": candidate_violations,
                            }
                        )
                        _mark_closed(state_hash)
                        continue

                branch_ctx = ContextManager.isolate(
                    node_ctx, purpose=f"tree_search:{step.name}:{current.node_id}"
                )
                if branch_ctx is not None and hasattr(branch_ctx, "tree_search_state"):
                    try:
                        setattr(branch_ctx, "tree_search_state", None)
                    except Exception:
                        pass

                eval_prompt = _build_prompt(
                    objective=objective,
                    path_summary=path_summary,
                    candidate=candidate,
                    purpose="evaluator",
                )

                try:
                    est_eval = _estimate_usage(core, evaluator, eval_prompt, branch_ctx)
                    _reserve_quota(quota, est_eval, limits)
                    eval_out = None
                    eval_sr: StepResult
                    if isinstance(evaluator, Pipeline):
                        exec_pipeline = getattr(core, "_execute_pipeline_via_policies", None)
                        if not callable(exec_pipeline):
                            raise TypeError("ExecutorCore missing _execute_pipeline_via_policies")
                        pr = await exec_pipeline(
                            evaluator,
                            eval_prompt,
                            branch_ctx,
                            resources,
                            limits,
                            context_setter=None,
                        )
                        eval_out = getattr(pr, "final_output", None)
                        if eval_out is None and pr.step_history:
                            eval_out = pr.step_history[-1].output
                        eval_sr = StepResult(
                            name=getattr(evaluator, "name", "evaluator"),
                            output=eval_out,
                            success=pr.success,
                            attempts=len(pr.step_history),
                            latency_s=0.0,
                            token_counts=int(getattr(pr, "total_tokens", 0) or 0),
                            cost_usd=float(getattr(pr, "total_cost_usd", 0.0) or 0.0),
                            step_history=list(pr.step_history),
                        )
                    else:
                        execute_step = getattr(core, "execute", None)
                        if not callable(execute_step):
                            raise TypeError("ExecutorCore missing execute")
                        outcome = await execute_step(
                            step=evaluator,
                            data=eval_prompt,
                            context=branch_ctx,
                            resources=resources,
                            limits=limits,
                            context_setter=None,
                            stream=False,
                            on_chunk=None,
                        )
                        unwrap_fn = getattr(core, "_unwrap_outcome_to_step_result", None)
                        if not callable(unwrap_fn):
                            raise TypeError("ExecutorCore missing _unwrap_outcome_to_step_result")
                        eval_sr = unwrap_fn(outcome, getattr(evaluator, "name", "evaluator"))
                        eval_out = eval_sr.output
                    actual_eval = UsageEstimate(
                        cost_usd=float(getattr(eval_sr, "cost_usd", 0.0) or 0.0),
                        tokens=int(getattr(eval_sr, "token_counts", 0) or 0),
                    )
                    _reconcile_quota(quota, est_eval, actual_eval, limits)
                except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
                    state.status = "paused"
                    _snapshot_state()
                    await _persist_frontier_state(status="paused")
                    raise

                step_history.append(eval_sr)
                total_cost += float(getattr(eval_sr, "cost_usd", 0.0) or 0.0)
                total_tokens += int(getattr(eval_sr, "token_counts", 0) or 0)

                eval_score, hard_fail, eval_meta = _score_evaluation(eval_out)
                h_cost = _compute_heuristic_cost(
                    eval_out,
                    score=eval_score,
                    hard_fail=hard_fail,
                    meta=eval_meta,
                )
                g_cost = _compute_cost(
                    step.cost_function,
                    candidate=candidate,
                    parent=current,
                    depth=current.depth + 1,
                    evaluation=eval_out,
                )
                f_cost = float("inf") if math.isinf(h_cost) else g_cost + h_cost
                _mark_closed(state_hash)

                node_id = f"n{state.next_node_id}"
                state.next_node_id += 1
                new_node = SearchNode(
                    node_id=node_id,
                    parent_id=current.node_id,
                    depth=current.depth + 1,
                    candidate=candidate,
                    output=candidate,
                    g_cost=g_cost,
                    h_cost=h_cost,
                    f_cost=f_cost,
                    state_hash=state_hash,
                    evaluation=eval_meta,
                    metadata={
                        "rubric_score": eval_score,
                        "hard_fail": hard_fail,
                    },
                )
                if branch_ctx is not None and isinstance(branch_ctx, PipelineContext):
                    new_node.attach_context(branch_ctx)
                state.nodes[node_id] = new_node
                state.open_set.append(node_id)
                expanded += 1

                best_node = _get_best_node(best_node, new_node, eval_score)
                state.best_node_id = best_node.node_id if best_node is not None else None
                _record_trace(
                    {
                        "event": "evaluated",
                        "node_id": node_id,
                        "score": eval_score,
                        "hard_fail": hard_fail,
                        "g_cost": g_cost,
                        "h_cost": h_cost,
                        "heuristic": eval_meta.get("heuristic_source"),
                    }
                )

                if _should_accept_goal(eval_score, step.goal_score_threshold, hard_fail):
                    winner = new_node
                    goal_reached = True
                    break

            state.expansions += 1
            if step.beam_width and step.beam_width > 0:
                best_open = state.sorted_open_nodes()[: step.beam_width]
                state.open_set = [n.node_id for n in best_open]
            _record_trace(
                {
                    "event": "expanded",
                    "node_id": current.node_id,
                    "candidates": len(candidates),
                    "accepted": expanded,
                }
            )
            _snapshot_state()
            await _persist_frontier_state()

            if goal_reached:
                break

        if winner is None and best_node is not None:
            winner = best_node
        heuristic_counts: dict[str, int] = {}
        for event in state.trace:
            if event.get("event") != "evaluated":
                continue
            key = event.get("heuristic") or "unknown"
            heuristic_counts[str(key)] = heuristic_counts.get(str(key), 0) + 1
        feedback: str | None = None
        if winner is None:
            feedback = "TreeSearchStep produced no viable candidates"
            failure_metadata: dict[str, object] = {
                "goal_reached": False,
                "iterations": state.iterations,
                "expansions": state.expansions,
            }
            if heuristic_counts:
                failure_metadata["heuristic_counts"] = heuristic_counts
            result = StepResult(
                name=step.name,
                success=False,
                output=None,
                attempts=state.expansions,
                latency_s=time.monotonic() - start_time,
                token_counts=total_tokens,
                cost_usd=total_cost,
                feedback=feedback,
                step_history=step_history,
                metadata_=failure_metadata,
            )
            state.status = "failed"
            _snapshot_state()
            return Failure(error=None, feedback=feedback, step_result=result)

        winner_ctx = winner.rehydrate_context(type(context))
        if context is not None and winner_ctx is not None:
            excluded = get_excluded_fields() | {
                "tree_search_state",
                "current_step",
                "current_step_index",
            }
            merged = safe_merge_context_updates(context, winner_ctx, excluded_fields=excluded)
            if merged is False:
                try:
                    merged_ctx = ContextManager.merge(context, winner_ctx)
                    if isinstance(merged_ctx, PipelineContext):
                        context = merged_ctx
                except Exception:
                    pass

        state.status = "completed" if (goal_reached or not step.require_goal) else "failed"
        _snapshot_state()

        output = winner.output if winner.output is not None else winner.candidate
        if step.require_goal and not goal_reached:
            feedback = "Goal threshold not reached"

        result_metadata: dict[str, object] = {
            "goal_reached": goal_reached,
            "iterations": state.iterations,
            "expansions": state.expansions,
            "best_node_id": winner.node_id,
            "best_score": winner.metadata.get("rubric_score"),
        }
        if heuristic_counts:
            result_metadata["heuristic_counts"] = heuristic_counts
        heuristic_source = None
        if winner.evaluation is not None:
            heuristic_source = winner.evaluation.get("heuristic_source")
        if heuristic_source is not None:
            result_metadata["best_heuristic_source"] = heuristic_source

        result = StepResult(
            name=step.name,
            success=(goal_reached or not step.require_goal),
            output=output,
            attempts=state.expansions,
            latency_s=time.monotonic() - start_time,
            token_counts=total_tokens,
            cost_usd=total_cost,
            feedback=feedback,
            step_history=step_history,
            metadata_=result_metadata,
        )
        try:
            if frame.context_setter is not None:
                pr: PipelineResult[BaseModel] = PipelineResult(
                    step_history=step_history,
                    total_cost_usd=total_cost,
                    total_tokens=total_tokens,
                    final_pipeline_context=context,
                )
                frame.context_setter(pr, context)
        except Exception:
            pass
        telemetry.logfire.debug(
            "TreeSearchStep completed",
            extra={
                "step": step.name,
                "goal_reached": goal_reached,
                "iterations": state.iterations,
            },
        )
        return Success(step_result=result)
