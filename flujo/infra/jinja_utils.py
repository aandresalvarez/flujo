from __future__ import annotations

from typing import Any


def create_jinja_environment(jinja_mod: Any) -> Any:
    """Create a Jinja environment preferring sandboxing and strict undefined.

    Args:
        jinja_mod: The imported jinja2 module.

    Returns:
        A configured Jinja environment instance.
    """
    try:
        try:
            from jinja2.sandbox import SandboxedEnvironment
        except Exception:
            SandboxedEnvironment = jinja_mod.Environment  # type: ignore[misc, assignment]
        return SandboxedEnvironment(undefined=jinja_mod.StrictUndefined, autoescape=False)
    except Exception:
        return jinja_mod.Environment(undefined=jinja_mod.StrictUndefined, autoescape=False)
