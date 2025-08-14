from __future__ import annotations

from typing import Any, Dict, Optional
import os
import json

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from .skill_registry import get_skill_registry
from importlib import import_module
import importlib.metadata as importlib_metadata


def _import_object(path: str) -> Any:
    if ":" in path:
        mod, attr = path.split(":", 1)
    else:
        parts = path.split(".")
        if len(parts) < 2:
            return import_module(path)
        mod, attr = ".".join(parts[:-1]), parts[-1]
    module = import_module(mod)
    return getattr(module, attr)


def load_skills_catalog(directory: str) -> None:
    """Load skills from a catalog file in the given directory.

    Supported files (first found wins): skills.yaml, skills.yml, skills.json
    Format YAML/JSON:
      echo-skill:
        path: "package.module:FactoryOrClass"
        description: "Echo agent"
    """
    candidates = [
        os.path.join(directory, "skills.yaml"),
        os.path.join(directory, "skills.yml"),
        os.path.join(directory, "skills.json"),
    ]
    path: Optional[str] = next((p for p in candidates if os.path.isfile(p)), None)
    if not path:
        return
    data: Dict[str, Dict[str, Any]]
    try:
        if path.endswith((".yaml", ".yml")) and yaml is not None:
            with open(path, "r") as f:
                raw = yaml.safe_load(f)
        else:
            with open(path, "r") as f:
                raw = json.load(f)
        if not isinstance(raw, dict):
            return
        data = raw  # type: ignore[assignment]
    except Exception:
        return

    reg = get_skill_registry()
    for skill_id, entry in data.items():
        try:
            obj = _import_object(entry.get("path", ""))
            reg.register(
                skill_id,
                obj,
                description=entry.get("description"),
                input_schema=entry.get("input_schema"),
                output_schema=entry.get("output_schema"),
            )
        except Exception:
            continue


def load_skills_entry_points(group: str = "flujo.skills") -> None:
    """Load skills from Python entry points (packaged plugins).

    Entry point value should be an import string (e.g., package.module:Factory).
    """
    try:
        eps = importlib_metadata.entry_points().select(group=group)  # type: ignore[attr-defined]
    except Exception:
        try:
            # Fallback for older Python/importlib_metadata API
            eps = importlib_metadata.entry_points().get(group, [])  # type: ignore[assignment]
        except Exception:
            eps = []  # type: ignore[assignment]

    reg = get_skill_registry()
    for ep in eps:
        try:
            obj = _import_object(ep.value)
            reg.register(ep.name, obj, description=f"entry_point:{group}")
        except Exception:
            continue
