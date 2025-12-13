from __future__ import annotations
# mypy: disable-error-code=arg-type

from typing import Any

from flujo.architect.states.common import (
    goto,
    make_transition_guard,
    skill_resolver,
    trace_next_state,
)
from flujo.domain.base_model import BaseModel as _BaseModel
from flujo.domain.dsl import Pipeline, Step
from flujo.type_definitions.common import JSONObject


async def _select_yaml_text(_x: Any = None, context: _BaseModel | None = None) -> str:
    try:
        yt = getattr(context, "yaml_text", "") if context is not None else ""
        return yt if isinstance(yt, str) else ""
    except Exception:
        return ""


def build_dry_run_offer_state() -> Pipeline[Any, Any]:
    """Optional dry-run offer; defaults to moving to Finalization."""

    async def _goto_final(
        _data: JSONObject | None = None, context: _BaseModel | None = None
    ) -> JSONObject:
        return await goto("Finalization", context=context)

    goto_final = Step.from_callable(
        _goto_final,
        name="GotoFinal",
        updates_context=True,
    )
    return (
        Pipeline.from_step(goto_final)
        >> Step.from_callable(
            make_transition_guard("Finalization"),
            name="Guard_Finalization_Offer",
            updates_context=True,
        )
        >> Step.from_callable(
            trace_next_state, name="TraceNextState_Finalization_Offer", updates_context=True
        )
    )


def build_dry_run_execution_state() -> Pipeline[Any, Any]:
    """Execute the pipeline in-memory before finalization."""
    reg = skill_resolver()
    try:
        dry_entry = reg.get("flujo.builtins.run_pipeline_in_memory")
        if dry_entry and isinstance(dry_entry, dict):
            _dry = dry_entry["factory"]()
            dryrun: Step[Any, Any] = Step.from_callable(
                _dry, name="DryRunInMemory", updates_context=False
            )
        else:

            async def _dry_fallback(*_a: Any, **_k: Any) -> JSONObject:
                return {"dry_run_result": {}}

            dryrun = Step.from_callable(_dry_fallback, name="DryRunInMemory")
    except Exception:

        async def _dry_fallback(*_a: Any, **_k: Any) -> JSONObject:
            return {"dry_run_result": {}}

        dryrun = Step.from_callable(_dry_fallback, name="DryRunInMemory")

    async def _goto_final(
        _data: JSONObject | None = None, context: _BaseModel | None = None
    ) -> JSONObject:
        return await goto("Finalization", context=context)

    goto_final = Step.from_callable(
        _goto_final,
        name="GotoFinal2",
        updates_context=True,
    )
    select_yaml = Step.from_callable(_select_yaml_text, name="SelectYAMLText")

    return (
        Pipeline.from_step(select_yaml)
        >> dryrun
        >> goto_final
        >> Step.from_callable(
            make_transition_guard("Finalization"),
            name="Guard_Finalization_Exec",
            updates_context=True,
        )
        >> Step.from_callable(
            trace_next_state, name="TraceNextState_Finalization_Exec", updates_context=True
        )
    )
