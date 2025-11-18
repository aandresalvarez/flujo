from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, MutableMapping

from .evidence import Badges, copy_badges, ensure_badges, update_badges


JsonDict = Dict[str, Any]


class VerifyResult(dict):
    """Thin dict subclass so Flujo style hooks can attach metadata."""

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)


def generic(context: Any) -> VerifyResult:
    badges = _consume_badges(context)
    domains = set(_iterable_from_context(context, "domains"))
    extra_support = 1 if len(domains) >= 2 else 0
    checklist_ok = bool(_resolve_path(context, ["output", "checklist_ok"], False))
    citations_ok = bool(_resolve_path(context, ["output", "has_required_citations"], True))
    updated = update_badges(
        badges,
        {
            "support": extra_support,
            "verify_hits": int(checklist_ok),
            "constraints_ok": citations_ok,
        },
    )
    _store_badges(context, updated)
    return VerifyResult(badges=updated.to_dict())


def citations(context: Any) -> VerifyResult:
    badges = _consume_badges(context)
    citation_count = int(_resolve_path(context, ["output", "citation_count"], 0))
    quality = float(_resolve_path(context, ["meta", "source_quality"], badges.quality_avg))
    updated = update_badges(
        badges,
        {
            "support": citation_count,
            "quality_avg": quality,
            "constraints_ok": citation_count > 0,
        },
    )
    _store_badges(context, updated)
    return VerifyResult(badges=updated.to_dict())


def run_pytests(context: Any) -> VerifyResult:
    badges = _consume_badges(context)
    pytest_pass = bool(_resolve_path(context, ["output", "pytest_passed"], False))
    test_runtime = float(_resolve_path(context, ["output", "pytest_runtime_s"], 0.0))
    updated = update_badges(
        badges,
        {
            "verify_hits": int(pytest_pass),
            "constraints_ok": pytest_pass,
            "cost_tokens": int(test_runtime * 10),
        },
    )
    _store_badges(context, updated)
    return VerifyResult(badges=updated.to_dict(), pytest_passed=pytest_pass)


def hitl_gate_or_nudge(context: Any) -> JsonDict:
    badges = _consume_badges(context)
    risk = float(_resolve_path(context, ["meta", "action_risk"], 0.4))
    uncertainty = float(_resolve_path(context, ["meta", "uncertainty"], 0.6))
    product = risk * uncertainty
    _append_trace(context, {"hitl_score": product, "risk": risk, "uncertainty": uncertainty})

    if _context_policy(context, "hard_gate"):
        decision = {"hitl": "gate"}
    elif product >= _hitl_threshold(context, default=0.35):
        decision = {"hitl": "nudge", "options": ["Option A", "Option B", "Default"]}
    else:
        decision = {"hitl": "none"}

    _store_badges(context, badges)
    return decision


def _consume_badges(context: Any) -> Badges:
    if isinstance(context, MutableMapping):
        return ensure_badges(context)
    existing = getattr(context, "badges", None)
    return copy_badges(existing)


def _store_badges(context: Any, badges: Badges) -> None:
    if isinstance(context, MutableMapping):
        context["badges"] = badges
    else:
        setattr(context, "badges", badges)


def _iterable_from_context(context: Any, key: str) -> Iterable[Any]:
    if isinstance(context, Mapping):
        value = context.get(key, [])
    else:
        value = getattr(context, key, [])
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return value
    return []


_MISSING = object()


def _resolve_path(context: Any, path: Iterable[Any], default: Any) -> Any:
    current: Any = context
    for part in path:
        if isinstance(current, Mapping):
            current = current.get(part, _MISSING)
        else:
            current = getattr(current, part, _MISSING)
        if current is _MISSING:
            return default
    return current


def _append_trace(context: Any, entry: JsonDict) -> None:
    trace = None
    if isinstance(context, Mapping):
        trace = context.get("trace")
    else:
        trace = getattr(context, "trace", None)
    if hasattr(trace, "add"):
        trace.add(entry)
    elif hasattr(trace, "append"):
        trace.append(entry)


def _context_policy(context: Any, policy_name: str) -> bool:
    policy_attr = None
    if hasattr(context, "policy"):
        policy_attr = context.policy
    elif isinstance(context, Mapping):
        policy_attr = context.get("policy")
    if callable(policy_attr):
        return bool(policy_attr(policy_name))
    return False


def _hitl_threshold(context: Any, default: float) -> float:
    cfg = None
    if isinstance(context, Mapping):
        cfg = context.get("cfg")
    else:
        cfg = getattr(context, "cfg", None)
    if cfg is None:
        return default
    threshold = _resolve_path(cfg, ["hitl", "threshold", "value"], default)
    try:
        return float(threshold)
    except (TypeError, ValueError):
        return default
