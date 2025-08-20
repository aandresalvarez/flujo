from __future__ import annotations

from typing import Any

from flujo.domain.dsl import Pipeline, Step


async def _emit_minimal_yaml(goal: str) -> dict[str, str]:
    """Return a minimal, valid Flujo YAML blueprint derived from the goal.

    The builder intentionally emits a conservative pipeline scaffold to keep
    CLI `create` flows fast and dependency-free.
    """
    safe_name = "generated_pipeline"
    try:
        g = (goal or "").strip()
        if g:
            # Simple normalization: keep alnum/space, collapse to underscores
            import re as _re

            norm = _re.sub(r"[^A-Za-z0-9\s]+", "", g)[:40].strip().lower()
            if norm:
                safe_name = ("_".join(norm.split()) or safe_name)[:40]
    except Exception:
        pass
    yaml_text = f'version: "0.1"\nname: {safe_name}\nsteps: []\n'
    return {"generated_yaml": yaml_text, "yaml_text": yaml_text}


def build_architect_pipeline() -> Pipeline[Any, Any]:
    """Return a minimal programmatic pipeline for the Architect flow.

    This replaces the legacy packaged YAML blueprint and is sufficient for
    CLI tests and local generation without networked LLMs.
    """
    step = Step.from_callable(_emit_minimal_yaml, name="GenerateYAML")
    return Pipeline.from_step(step)
