from __future__ import annotations

from typing import Any
from flujo.type_definitions.common import JSONObject

from flujo.architect.states.common import goto, skill_resolver
from flujo.domain.base_model import BaseModel as _BaseModel
from flujo.domain.dsl import Pipeline, Step
from flujo.exceptions import InfiniteRedirectError, PausedException, PipelineAbortSignal


async def approval_noop(x: str, *, context: _BaseModel | None = None) -> str:
    """No-op PlanApproval step for minimal pipelines."""
    try:
        if context is not None and hasattr(context, "plan_approved"):
            setattr(context, "plan_approved", True)
    except Exception:
        pass
    return x


async def _plan_approval_runner(_x: Any = None, *, context: _BaseModel | None = None) -> JSONObject:
    """
    Decide whether to approve the plan. In non-interactive mode, this always defaults to approved.
    In interactive mode, it can prompt the user. This prevents infinite loops caused by
    stale 'plan_approved: False' flags in the context.
    """
    hitl = False
    noni = False
    try:
        if context is not None:
            hitl = bool(getattr(context, "hitl_enabled", False))
            noni = bool(getattr(context, "non_interactive", False))
    except Exception:
        pass

    approved = True  # Default to approved, respecting idempotency.

    if hitl and not noni:
        try:
            reg = skill_resolver()
            ask_entry = reg.get("flujo.builtins.ask_user") or {}
            chk_entry = reg.get("flujo.builtins.check_user_confirmation") or {}
            ask_factory = ask_entry.get("factory") if ask_entry else None
            chk_factory = chk_entry.get("factory") if chk_entry else None
            _ask = ask_factory() if ask_factory is not None else None
            _chk = chk_factory() if chk_factory is not None else None
            if _ask is not None and _chk is not None:
                resp = await _ask(question="Does this plan look correct? (Y/n)")
                key = await _chk(user_input=str(resp))
                approved = str(key).strip().lower() == "approved"
        except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
            # Preserve orchestration control-flow
            raise
        except Exception:
            approved = True

    nxt = await goto("ParameterCollection" if approved else "Refinement", context=context)
    return {"plan_approved": approved, **nxt}


def build_plan_approval_state() -> Pipeline[Any, Any]:
    """Interactive or automatic plan approval decision."""
    return Pipeline.from_step(
        Step.from_callable(_plan_approval_runner, name="PlanApproval", updates_context=True)
    )
