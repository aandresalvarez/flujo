from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel as PydanticBaseModel
from flujo.domain.models import BaseModel as DomainBaseModel

from flujo.infra.skills_catalog import load_skills_catalog, load_skills_entry_points
from flujo.infra.skill_registry import get_skill_registry
from flujo.domain.agent_protocol import AsyncAgentProtocol


class DiscoverSkillsAgent(AsyncAgentProtocol[Any, Dict[str, Any]]):
    """Builtin agent that discovers available skills and exposes them to context.

    - Loads skills from a local catalog (skills.yaml/skills.json) and Python entry points.
    - Returns a structure suitable for LLM tool matching steps.
    """

    def __init__(self, directory: Optional[str] = None) -> None:
        self.directory = directory or "."

    async def run(self, data: Any, **kwargs: Any) -> Dict[str, Any]:
        # Best-effort: load catalog + packaged entry points
        try:
            load_skills_catalog(self.directory)
            load_skills_entry_points()
        except Exception:
            # Non-fatal; continue with whatever is registered
            pass

        # Collect a public view of registered skills
        skills: List[Dict[str, Any]] = []
        try:
            reg = get_skill_registry()
            entries = getattr(reg, "_entries", {})  # Access internal map read-only
            for sid, meta in entries.items():
                skills.append(
                    {
                        "id": sid,
                        "description": meta.get("description"),
                        "input_schema": meta.get("input_schema"),
                    }
                )
        except Exception:
            # If registry access fails, return empty list
            skills = []

        return {"available_skills": skills}


# --- Adapter: extract decomposed steps into a flat context key ---
async def extract_decomposed_steps(
    decomposition: Any, *, output_key: str = "prepared_steps_for_mapping"
) -> Dict[str, Any]:
    """Adapter to extract a list of step dicts from the decomposer output.

    Returns a dict so that `updates_context: true` can merge it into the pipeline context.
    """
    steps: List[Dict[str, Any]] = []
    try:
        # Handle pydantic models with .model_dump()
        if isinstance(decomposition, PydanticBaseModel):
            try:
                raw = decomposition.model_dump()
            except Exception:
                raw = {}
            if isinstance(raw, dict):
                cand = raw.get("steps")
                if isinstance(cand, list):
                    steps = [x for x in cand if isinstance(x, dict)]
        # Handle plain dict
        elif isinstance(decomposition, dict):
            cand = decomposition.get("steps")
            if isinstance(cand, list):
                steps = [x for x in cand if isinstance(x, dict)]
        # Handle object attribute access
        else:
            cand = getattr(decomposition, "steps", None)
            if isinstance(cand, list):
                steps = [x for x in cand if isinstance(x, dict)]
    except Exception:
        steps = []

    return {output_key: steps}


# --- Adapter: extract YAML text from writer output ---
async def extract_yaml_text(writer_output: Any) -> str:
    """Extract the YAML text field from a YamlWriter agent's output.

    Handles multiple shapes gracefully:
    - pydantic-like object with attribute 'yaml_text' or 'generated_yaml'
    - dict payload with keys 'yaml_text' or 'generated_yaml'
    - falls back to str() representation
    """
    try:
        # Attribute access (pydantic model or simple object)
        if hasattr(writer_output, "yaml_text"):
            val = getattr(writer_output, "yaml_text")
            if isinstance(val, (str, bytes)):
                return val.decode() if isinstance(val, bytes) else val
        if hasattr(writer_output, "generated_yaml"):
            val = getattr(writer_output, "generated_yaml")
            if isinstance(val, (str, bytes)):
                return val.decode() if isinstance(val, bytes) else val

        # Mapping access
        if isinstance(writer_output, dict):
            if "yaml_text" in writer_output:
                val = writer_output["yaml_text"]
                if isinstance(val, (str, bytes)):
                    return val.decode() if isinstance(val, bytes) else val
            if "generated_yaml" in writer_output:
                val = writer_output["generated_yaml"]
                if isinstance(val, (str, bytes)):
                    return val.decode() if isinstance(val, bytes) else val
    except Exception:
        pass

    return str(writer_output)


# --- Adapter: turn ValidationReport into a boolean flag on context ---
async def validation_report_to_flag(report: Any) -> Dict[str, Any]:
    """Return a dict with yaml_is_valid based on a ValidationReport-like input."""
    try:
        if isinstance(report, dict):
            val = bool(report.get("is_valid", False) or report.get("yaml_is_valid", False))
        else:
            val = bool(getattr(report, "is_valid", False))
    except Exception:
        val = False
    return {"yaml_is_valid": val}


def exit_when_yaml_valid(_out: Any, context: Any | None) -> bool:
    """Exit condition for LoopStep: stop when context.yaml_is_valid is true."""
    try:
        return bool(getattr(context, "yaml_is_valid", False))
    except Exception:
        try:
            if isinstance(context, dict):
                return bool(context.get("yaml_is_valid", False))
        except Exception:
            pass
        return False


def _register_builtins() -> None:
    """Register builtin skills with the global registry."""
    try:
        reg = get_skill_registry()
        # Factory accepts params to match YAML 'agent: { id: ..., params: {...} }'
        reg.register(
            "flujo.builtins.discover_skills",
            lambda directory=".": DiscoverSkillsAgent(directory=directory),
            description="Discover local and packaged skills; returns available_skills list.",
        )
        # Adapter function: return the async callable without invoking it
        # Loader will call this factory with params (none by default) and expect an agent object
        reg.register(
            "flujo.builtins.extract_decomposed_steps",
            # Factory returns the coroutine function itself so Step.from_callable can wrap it
            lambda **_params: extract_decomposed_steps,
            description=(
                "Extract list of step dicts from decomposer output into 'prepared_steps_for_mapping'"
            ),
        )
        # Adapter extractor for YAML string
        reg.register(
            "flujo.builtins.extract_yaml_text",
            lambda **_params: extract_yaml_text,
            description="Extract YAML string from YamlWriter output object or dict.",
        )
        reg.register(
            "flujo.builtins.validation_report_to_flag",
            lambda **_params: validation_report_to_flag,
            description="Map validation report to {'yaml_is_valid': bool} and update context.",
        )

        # Aggregator: combine mapped results with goal and (optional) skills
        async def aggregate_plan(
            mapped_step_results: Any, *, context: DomainBaseModel | None = None
        ) -> Dict[str, Any]:
            try:
                user_goal = getattr(context, "user_goal", None) or getattr(
                    context, "initial_prompt", None
                )
            except Exception:
                user_goal = None
            # Normalize list of results into list of dicts
            plans: List[Dict[str, Any]] = []
            try:
                if isinstance(mapped_step_results, list):
                    for item in mapped_step_results:
                        if isinstance(item, dict):
                            plans.append(item)
                        elif hasattr(item, "model_dump") and callable(getattr(item, "model_dump")):
                            try:
                                plans.append(item.model_dump())
                            except Exception:
                                pass
                        else:
                            try:
                                plans.append(dict(item))
                            except Exception:
                                pass
            except Exception:
                plans = []

            skills: List[Dict[str, Any]] = []
            try:
                maybe = getattr(context, "available_skills", None)
                if isinstance(maybe, list):
                    skills = [x for x in maybe if isinstance(x, dict)]
            except Exception:
                pass

            return {
                "user_goal": user_goal or "",
                "step_plans": plans,
                "available_skills": skills,
            }

        reg.register(
            "flujo.builtins.aggregate_plan",
            lambda **_params: aggregate_plan,
            description="Aggregate mapped tool decisions and goal for YAML writer.",
        )
    except Exception:
        # Registration failures should not break import
        pass


# Register on import so CLI/YAML resolution can find it
_register_builtins()
