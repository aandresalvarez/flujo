from __future__ import annotations

from typing import Any, Callable, Optional
import os
import json

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from .skill_registry import get_skill_registry
from importlib import import_module
import importlib.metadata as importlib_metadata
from flujo.type_definitions.common import JSONObject


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
    data: dict[str, JSONObject]
    try:
        if path.endswith((".yaml", ".yml")) and yaml is not None:
            with open(path, "r") as f:
                raw = yaml.safe_load(f)
        else:
            with open(path, "r") as f:
                raw = json.load(f)
        if not isinstance(raw, dict):
            return
        # Raw must be a mapping of skill_id -> attributes; enforce type at runtime
        tmp: dict[str, JSONObject] = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                tmp[str(k)] = dict(v)
        data = tmp
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
                capabilities=entry.get("capabilities"),
                safety_level=entry.get("safety_level"),
                auth_required=entry.get("auth_required"),
                auth_scope=entry.get("auth_scope"),
                side_effects=entry.get("side_effects"),
                # Accept FSD-020 alias
                arg_schema=entry.get("arg_schema"),
            )
        except Exception:
            continue


def load_skills_entry_points(group: str = "flujo.skills") -> None:
    """Load skills from Python entry points (packaged plugins).

    Entry point value should be an import string (e.g., package.module:Factory).
    """
    eps_list = list(importlib_metadata.entry_points().select(group=group))

    reg = get_skill_registry()
    for ep in eps_list:
        try:
            obj = _import_object(ep.value)
            reg.register(ep.name, obj, description=f"entry_point:{group}")
        except Exception:
            continue


# Wire domain-level skills discovery interface to avoid infra imports in domain logic
set_default_skills_discovery_fn: Optional[Callable[[Any], None]]
try:  # pragma: no cover - import guard
    from flujo.domain.interfaces import (
        set_default_skills_discovery as _set_default_skills_discovery_fn,
    )

    set_default_skills_discovery_fn = _set_default_skills_discovery_fn
except Exception:  # pragma: no cover - defensive fallback
    set_default_skills_discovery_fn = None


class _SkillsDiscoveryAdapter:
    def load_catalog(self, base_dir: str) -> None:  # pragma: no cover - simple delegation
        load_skills_catalog(base_dir)

    def load_entry_points(self) -> None:  # pragma: no cover - simple delegation
        load_skills_entry_points()


if set_default_skills_discovery_fn is not None:  # pragma: no cover - simple wiring
    try:
        set_default_skills_discovery_fn(_SkillsDiscoveryAdapter())
    except Exception:
        pass
