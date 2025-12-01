from __future__ import annotations

from typing import Any, Optional

from flujo.type_definitions.common import JSONObject

from flujo.architect.states.common import normalize_name_from_goal
from flujo.architect.states.generation import (
    emit_minimal_yaml,
    generate_yaml_from_plan,
    generate_yaml_from_tool_selections,
)
from flujo.domain.base_model import BaseModel as _BaseModel
from flujo.domain.dsl import Pipeline, Step
from flujo.exceptions import InfiniteRedirectError, PausedException, PipelineAbortSignal


async def _finalize(_x: Any = None, *, context: _BaseModel | None = None) -> JSONObject:
    """Ensure final YAML is present and return it as the step output."""
    yaml_text: Optional[str] = None
    try:
        if context is not None:
            try:
                gen_yaml_struct = getattr(context, "generated_yaml_structured", None)
                if gen_yaml_struct is not None and hasattr(gen_yaml_struct, "generated_yaml"):
                    yaml_text = getattr(gen_yaml_struct, "generated_yaml")
            except Exception:
                pass

            if not yaml_text:
                try:
                    yaml_text = getattr(context, "generated_yaml", None) or getattr(
                        context, "yaml_text", None
                    )
                except Exception:
                    yaml_text = None

            if (not isinstance(yaml_text, str) or not yaml_text.strip()) and hasattr(
                context, "tool_selections"
            ):
                try:
                    gen = await generate_yaml_from_tool_selections(None, context=context)
                    if isinstance(gen, dict):
                        cand = gen.get("yaml_text") or gen.get("generated_yaml")
                        if isinstance(cand, str) and cand.strip():
                            yaml_text = cand
                except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
                    raise
                except Exception:
                    pass

            if not isinstance(yaml_text, str) or not yaml_text.strip():
                try:
                    gen2 = await generate_yaml_from_plan(None, context=context)
                    if isinstance(gen2, dict):
                        cand2 = gen2.get("yaml_text") or gen2.get("generated_yaml")
                        if isinstance(cand2, str) and cand2.strip():
                            yaml_text = cand2
                except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
                    raise
                except Exception:
                    pass

        if not isinstance(yaml_text, str) or not yaml_text.strip():
            try:
                goal = getattr(context, "user_goal", None) if context is not None else None
            except Exception:
                goal = None
            minimal = await emit_minimal_yaml(str(goal or "pipeline"))
            yaml_text = minimal.get("generated_yaml")
            try:
                if isinstance(yaml_text, str) and "steps: []" in yaml_text:
                    name = normalize_name_from_goal(str(goal) if goal is not None else None)
                    import yaml as _yaml

                    step_dict = {
                        "kind": "step",
                        "name": "Echo Input",
                        "agent": {"id": "flujo.builtins.stringify", "params": {}},
                    }
                    block = _yaml.safe_dump(step_dict, sort_keys=False).strip()
                    steps_block = "\n".join(
                        [
                            "- " + line if i == 0 else "  " + line
                            for i, line in enumerate(block.splitlines())
                        ]
                    )
                    yaml_text = f'\nversion: "0.1"\nname: {name}\nsteps:\n{steps_block}\n'
                    try:
                        setattr(context, "generated_yaml", yaml_text)
                        setattr(context, "yaml_text", yaml_text)
                    except Exception:
                        pass
            except Exception:
                pass
    except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
        raise
    except Exception:
        yaml_text = 'version: "0.1"\nname: fallback_pipeline\nsteps: []\n'

    if isinstance(yaml_text, str):
        empty_steps = "steps: []" in yaml_text
        if empty_steps:
            goal_value = getattr(context, "user_goal", None)
            name = normalize_name_from_goal(str(goal_value or "pipeline"))
            import yaml as _yaml

            step_dict = {
                "kind": "step",
                "name": "Echo Input",
                "agent": {"id": "flujo.builtins.stringify", "params": {}},
            }
            block = _yaml.safe_dump(step_dict, sort_keys=False).strip()
            steps_block = "\n".join(
                [
                    "- " + line if i == 0 else "  " + line
                    for i, line in enumerate(block.splitlines())
                ]
            )
            yaml_text = f'\nversion: "0.1"\nname: {name}\nsteps:\n{steps_block}\n'
            try:
                setattr(context, "generated_yaml", yaml_text)
                setattr(context, "yaml_text", yaml_text)
            except Exception:
                pass

    try:
        if context is not None:
            if hasattr(context, "yaml_text"):
                setattr(context, "yaml_text", yaml_text)
            if hasattr(context, "generated_yaml"):
                setattr(context, "generated_yaml", yaml_text)
    except Exception:
        pass
    return {"generated_yaml": yaml_text, "yaml_text": yaml_text}


def build_finalization_state() -> Pipeline[Any, Any]:
    fin = Step.from_callable(_finalize, name="Finalize", updates_context=True)
    return Pipeline.from_step(fin)


def build_failure_state() -> Pipeline[Any, Any]:
    async def _failure_step(*_a: Any, **_k: Any) -> JSONObject:
        return {}

    return Pipeline.from_step(Step.from_callable(_failure_step, name="Failure"))
