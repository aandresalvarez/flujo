from __future__ import annotations

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel as PydanticBaseModel
from flujo.domain.models import BaseModel as DomainBaseModel

from flujo.infra.skills_catalog import load_skills_catalog, load_skills_entry_points
from flujo.infra.skill_registry import get_skill_registry
from flujo.domain.agent_protocol import AsyncAgentProtocol
from .agents.wrapper import make_agent_async

# Optional dependency: duckduckgo-search (installed via extras: flujo[skills])
# Prefer async client if available; fall back to sync DDGS via thread pool.
_DDGSAsync: Optional[Type[Any]] = None
_DDGS_CLASS: Optional[Type[Any]] = None
try:  # pragma: no cover - optional dependency
    import duckduckgo_search as _dds

    _async = getattr(_dds, "AsyncDDGS", None)
    _sync = getattr(_dds, "DDGS", None)
    if _async is not None:
        _DDGSAsync = _async
    if _sync is not None:
        _DDGS_CLASS = _sync
except Exception:
    _DDGSAsync = None
    _DDGS_CLASS = None


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


# --- HITL helper: interpret user confirmation into branch key ---
async def check_user_confirmation(user_input: str) -> str:
    """Map free-form user input to a conditional branch key.

    Returns:
        "approved" for affirmative ("y", "yes", empty/whitespace), otherwise "denied".
    """
    try:
        text = "" if user_input is None else str(user_input)
    except Exception:
        text = ""
    norm = text.strip().lower()
    if norm == "":
        return "approved"
    affirmatives = {"y", "yes"}
    if norm in affirmatives:
        return "approved"
    return "denied"


# --- In-memory YAML validation skill ---
async def validate_yaml(yaml_text: str, base_dir: Optional[str] = None) -> Any:
    """Validate a YAML blueprint string and return a ValidationReport.

    Never raises for invalid YAML; returns a report with an error finding instead.
    """
    try:
        from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml

        pipeline = load_pipeline_blueprint_from_yaml(yaml_text, base_dir=base_dir)
        # Let the pipeline perform its deeper graph validation
        if pipeline is not None:
            return pipeline.validate_graph()
        else:
            # Handle case where pipeline loading returns None
            from flujo.domain.pipeline_validation import (
                ValidationReport as _VR,
                ValidationFinding as _VF,
            )

            return _VR(
                errors=[
                    _VF(
                        rule_id="YAML-LOAD",
                        severity="error",
                        message="Pipeline loading returned None",
                    ),
                ],
                warnings=[],
            )
    except Exception as e:
        try:
            # Construct a report capturing the parse/compile error
            from flujo.domain.pipeline_validation import (
                ValidationReport as _VR,
                ValidationFinding as _VF,
            )

            return _VR(
                errors=[
                    _VF(rule_id="YAML-PARSE", severity="error", message=str(e)),
                ],
                warnings=[],
            )
        except Exception:
            # Absolute fallback: minimal dict compatible with adapters/predicates
            return {"is_valid": False, "errors": [str(e)], "warnings": []}


# --- Passthrough adapter (identity) ---
async def passthrough(x: Any) -> Any:
    """Return the input unchanged (identity)."""
    return x


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
        # Human-in-the-loop confirmation interpreter
        reg.register(
            "flujo.builtins.check_user_confirmation",
            lambda **_params: check_user_confirmation,
            description=(
                "Interpret user input as 'approved' when affirmative (y/yes/empty), else 'denied'."
            ),
            arg_schema={
                "type": "object",
                "properties": {"user_input": {"type": "string"}},
                "required": ["user_input"],
            },
            side_effects=False,
        )
        # Identity adapter useful in conditional valid branches
        reg.register(
            "flujo.builtins.passthrough",
            lambda **_params: passthrough,
            description="Identity adapter that returns input unchanged.",
            side_effects=False,
        )
        # In-memory YAML validation that returns a ValidationReport and never raises on invalid YAML
        reg.register(
            "flujo.builtins.validate_yaml",
            lambda **_params: validate_yaml,
            description=(
                "Validate YAML blueprint text in-memory; returns ValidationReport without raising."
            ),
            arg_schema={
                "type": "object",
                "properties": {
                    "yaml_text": {"type": "string"},
                    "base_dir": {"type": "string"},
                },
                "required": ["yaml_text"],
            },
            side_effects=False,
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

        # --- Killer Demo: web_search ---
        async def web_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
            """Perform a DuckDuckGo web search (top N simplified results).

            Returns a list of {title, link, snippet} dicts.
            """
            if _DDGSAsync is None and _DDGS_CLASS is None:
                # Graceful degrade if optional dependency not installed
                return []

            results: List[Dict[str, Any]] = []
            try:
                if _DDGSAsync is not None:
                    # Use async client when available
                    async with _DDGSAsync() as ddgs:
                        async for r in ddgs.text(query, max_results=max_results):
                            if isinstance(r, dict):
                                results.append(r)
                else:
                    # Use DDGS in a thread pool since sync
                    import asyncio
                    from concurrent.futures import ThreadPoolExecutor

                    def _search_sync() -> List[Dict[str, Any]]:
                        assert _DDGS_CLASS is not None
                        ddgs = _DDGS_CLASS()
                        search_results: List[Dict[str, Any]] = []
                        for r in ddgs.text(query, max_results=max_results):
                            if isinstance(r, dict):
                                search_results.append(r)
                        return search_results

                    loop = asyncio.get_event_loop()
                    with ThreadPoolExecutor() as executor:
                        results = await loop.run_in_executor(executor, _search_sync)
            except Exception:
                # Non-fatal: return empty results on any search error
                return []

            simplified = [
                {
                    "title": item.get("title"),
                    "link": item.get("href"),
                    "snippet": item.get("body"),
                }
                for item in results
            ]
            return simplified

        reg.register(
            "flujo.builtins.web_search",
            lambda **_params: web_search,
            description=(
                "Performs a web search and returns the top results (titles, links, snippets)."
            ),
            arg_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
            side_effects=False,
        )

        # --- Killer Demo: extract_from_text ---
        async def extract_from_text(
            text: str,
            schema: Dict[str, Any],
            *,
            model: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Extract structured data from unstructured text using an LLM.

            The JSON schema is used as instruction; output is a dict.
            """
            # Default lightweight model consistent with examples
            chosen_model = model or "openai:gpt-4o-mini"

            system_prompt = (
                "You extract structured data from text.\n"
                "Return only valid JSON matching the provided JSON Schema.\n"
                "Do not include prose, backticks, or explanations.\n"
            )

            # Compose a single input string; the wrapper handles retries/repair
            input_payload = (
                "JSON_SCHEMA:\n"
                f"{schema}\n\n"
                "TEXT:\n"
                f"{text}\n\n"
                "Respond with JSON that validates against JSON_SCHEMA."
            )

            agent = make_agent_async(
                model=chosen_model,
                system_prompt=system_prompt,
                output_type=Dict[str, Any],
                max_retries=2,
                auto_repair=True,
            )

            result = await agent.run(input_payload)
            # The wrapper returns processed content; ensure it's a dict
            return result if isinstance(result, dict) else {"result": result}

        reg.register(
            "flujo.builtins.extract_from_text",
            lambda **_params: extract_from_text,
            description=(
                "Extracts structured data from text based on a provided JSON schema using an LLM."
            ),
            arg_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "schema": {"type": "object"},
                    "model": {"type": "string"},
                },
                "required": ["text", "schema"],
            },
            side_effects=False,
        )
    except Exception:
        # Registration failures should not break import
        pass


# Register on import so CLI/YAML resolution can find it
_register_builtins()
