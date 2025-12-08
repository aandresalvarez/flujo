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
            from jinja2.sandbox import SandboxedEnvironment as _SandboxedEnv

            env_cls: Any = _SandboxedEnv
        except Exception:
            env_cls = getattr(jinja_mod, "Environment", None)
        if env_cls is None:
            raise RuntimeError("Jinja environment class not available")
        return env_cls(undefined=jinja_mod.StrictUndefined, autoescape=False)
    except Exception:
        return jinja_mod.Environment(undefined=jinja_mod.StrictUndefined, autoescape=False)
