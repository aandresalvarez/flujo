from __future__ import annotations

import os as _os
from typing import Any

from flujo.architect.states import (
    approval_noop,
    build_dry_run_execution_state,
    build_dry_run_offer_state,
    build_failure_state,
    build_finalization_state,
    build_gathering_state,
    build_generation_state,
    build_goal_clarification_state,
    build_parameter_collection_state,
    build_plan_approval_state,
    build_planning_state,
    build_refinement_state,
    build_validation_state,
    emit_minimal_yaml,
    generate_yaml_from_tool_selections,
    match_one_tool,
    prepare_for_map,
    run_planner_agent,
    goto as _goto_state,
)
from flujo.domain.dsl import Pipeline, Step
from flujo.domain.dsl.state_machine import StateMachineStep
from flujo.domain.interfaces import get_config_provider

# Ensure core builtin skills are registered for heuristics
try:  # pragma: no cover - best-effort
    import flujo.builtins as _builtins  # noqa: F401
except Exception:
    pass

# Backwards compatibility for legacy imports in tests
_emit_minimal_yaml = emit_minimal_yaml
_generate_yaml_from_tool_selections = generate_yaml_from_tool_selections
_run_planner_agent = run_planner_agent
_approval_noop = approval_noop
_prepare_for_map = prepare_for_map
_match_one_tool = match_one_tool
_goto = _goto_state


def _build_state_machine_pipeline() -> Pipeline[Any, Any]:
    """Programmatically build the full Architect state machine."""
    gathering = build_gathering_state()
    goal_pipe = build_goal_clarification_state()
    plan_pipe = build_planning_state()
    approval_pipe = build_plan_approval_state()
    refine_pipe = build_refinement_state()
    params_pipe = build_parameter_collection_state()
    gen_pipeline = build_generation_state()
    validation_pipe = build_validation_state()
    dry_offer_pipe = build_dry_run_offer_state()
    dry_exec_pipe = build_dry_run_execution_state()
    fin_pipeline = build_finalization_state()
    failure_pipeline = build_failure_state()

    sm = StateMachineStep(
        name="Architect",
        states={
            "GatheringContext": gathering,
            "GoalClarification": goal_pipe,
            "Planning": plan_pipe,
            "PlanApproval": approval_pipe,
            "Refinement": refine_pipe,
            "ParameterCollection": params_pipe,
            "Generation": gen_pipeline,
            "Validation": validation_pipe,
            "DryRunOffer": dry_offer_pipe,
            "DryRunExecution": dry_exec_pipe,
            "Finalization": fin_pipeline,
            "Failure": failure_pipeline,
        },
        start_state="GatheringContext",
        end_states=["Finalization", "Failure"],
    )
    return Pipeline.from_step(sm)


def build_architect_pipeline() -> Pipeline[Any, Any]:
    """Return the Architect pipeline object.

    Behavior:
    - If test/CI overrides are enabled (``FLUJO_TEST_MODE`` or ``FLUJO_ARCHITECT_IGNORE_CONFIG``)
      → **always** use the minimal pipeline to keep perf tests deterministic.
    - Else if ``FLUJO_ARCHITECT_STATE_MACHINE`` is truthy → enable state machine.
    - Else, honor ``flujo.toml``: if ``[architect].state_machine_default = true`` → state machine.
    - Else → minimal, single-step generator (unit-test friendly default).

    This respects the team guide: use ConfigManager (not direct file reads) and
    allow explicit environment overrides for reproducible tests.
    """

    def _truthy(val: str | None) -> bool:
        v = (val or "").strip().lower()
        return v in {"1", "true", "yes", "on"}

    test_mode = _truthy(_os.environ.get("FLUJO_TEST_MODE"))
    ignore_cfg = _truthy(_os.environ.get("FLUJO_ARCHITECT_IGNORE_CONFIG"))

    if _truthy(_os.environ.get("FLUJO_ARCHITECT_STATE_MACHINE")):
        return _build_state_machine_pipeline()

    if test_mode or ignore_cfg:
        approval = Step.from_callable(_approval_noop, name="PlanApproval", updates_context=True)
        gen = Step.from_callable(_emit_minimal_yaml, name="GenerateYAML", updates_context=True)
        return Pipeline.from_step(approval) >> gen

    try:
        cfg = get_config_provider().load_config()
        arch = getattr(cfg, "architect", None) if cfg is not None else None
        if arch and bool(getattr(arch, "state_machine_default", False)):
            return _build_state_machine_pipeline()
    except Exception:
        pass

    gen = Step.from_callable(_emit_minimal_yaml, name="GenerateYAML", updates_context=True)
    return Pipeline.from_step(gen)
