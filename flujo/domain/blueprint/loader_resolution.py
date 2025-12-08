from __future__ import annotations

import importlib
import re
from typing import Any, List, Optional, Tuple, Union

from flujo.exceptions import InfiniteRedirectError, PausedException, PipelineAbortSignal
from ..interfaces import get_config_provider, get_skill_resolver
from .loader_models import BlueprintError
from flujo.type_definitions.common import JSONObject

_skills_base_dir_stack: list[str] = []


def _push_skills_base_dir(dir_path: Optional[str]) -> None:
    if dir_path:
        try:
            _skills_base_dir_stack.append(dir_path)
        except Exception:
            pass


def _pop_skills_base_dir() -> None:
    try:
        if _skills_base_dir_stack:
            _skills_base_dir_stack.pop()
    except Exception:
        pass


def _current_skills_base_dir() -> Optional[str]:
    try:
        return _skills_base_dir_stack[-1] if _skills_base_dir_stack else None
    except Exception:
        return None


def _import_object(path: str) -> Any:
    """Import an object from 'module:attr' or 'module.attr' path with allow-list enforcement."""
    module_name: str
    attr_name: Optional[str] = None

    if ".." in path or "/" in path or "\\" in path or path.strip().startswith("."):
        raise BlueprintError("Invalid import path: traversal or illegal characters are not allowed")

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\.]*(:[A-Za-z_][A-Za-z0-9_]*)?", path):
        raise BlueprintError("Invalid import path format")

    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        parts = path.split(".")
        if len(parts) < 2:
            module_name = path
            attr_name = None
        else:
            module_name, attr_name = ".".join(parts[:-1]), parts[-1]

    allowed: Optional[list[str]] = None
    try:
        cfg = get_config_provider().load_config()
        if cfg is not None:
            settings = getattr(cfg, "settings", None)
            if settings is not None:
                allowed = getattr(settings, "blueprint_allowed_imports", None)
            if allowed is None:
                allowed = getattr(cfg, "blueprint_allowed_imports", None)
        if allowed is not None and not isinstance(allowed, list):
            raise BlueprintError(
                "Configuration 'blueprint_allowed_imports' must be a list of module prefixes."
            )
    except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
        raise
    except BlueprintError:
        raise
    except Exception as exc:
        raise BlueprintError(
            "Failed to verify allowed imports from configuration; refusing to import modules from YAML."
        ) from exc

    # Default-deny: require explicit allow-list in configuration.
    if allowed is None:
        raise BlueprintError(
            "Blueprint imports are disallowed by default. Configure 'blueprint_allowed_imports' in configuration."
        )

    normalized_allowed = [a.strip() for a in allowed if isinstance(a, str) and a.strip()]
    allow_all = any(a == "*" for a in normalized_allowed)

    if not allow_all and not any(
        module_name == a or module_name.startswith(f"{a}.") for a in normalized_allowed
    ):
        raise BlueprintError(
            f"Import of module '{module_name}' is not allowed. Configure 'blueprint_allowed_imports' in configuration."
        )

    if module_name == "skills" or module_name.startswith("skills."):
        return _import_child_skill_module(module_name, attr_name)

    module = importlib.import_module(module_name)
    return getattr(module, attr_name) if attr_name else module


def _import_child_skill_module(module_name: str, attr_name: Optional[str]) -> Any:
    base_dir = _current_skills_base_dir()
    if not base_dir:
        raise BlueprintError(f"Unable to locate module '{module_name}' under current skills base")
    try:
        import importlib.util as _iu
        import hashlib as _hashlib
        import os as _os
        import sys as _sys
        import types as _types

        tail = module_name.split(".", 1)[1] if module_name != "skills" else ""
        parts = [p for p in tail.split(".") if p]
        skills_root = _os.path.join(base_dir, "skills")

        mod_path = _os.path.join(skills_root, *parts)
        is_package = False
        py_path = mod_path + (".py" if parts else _os.sep + "__init__.py")
        if parts and not _os.path.exists(py_path):
            pkg_init = _os.path.join(skills_root, *parts, "__init__.py")
            if _os.path.exists(pkg_init):
                py_path = pkg_init
                is_package = True
        elif not parts:
            is_package = True
        if not _os.path.exists(py_path):
            raise BlueprintError(
                f"Unable to locate module '{module_name}' under '{base_dir}/skills'"
            )

        token = _hashlib.sha1(str(base_dir).encode("utf-8")).hexdigest()[:10]
        root_pkg = f"__flujo_import__{token}"
        skills_pkg = f"{root_pkg}.skills"

        if root_pkg not in _sys.modules:
            root_mod = _types.ModuleType(root_pkg)
            root_mod.__path__ = []
            root_mod.__package__ = root_pkg
            _sys.modules[root_pkg] = root_mod
        if skills_pkg not in _sys.modules:
            skills_mod = _types.ModuleType(skills_pkg)
            skills_mod.__path__ = [skills_root]
            skills_mod.__package__ = skills_pkg
            _sys.modules[skills_pkg] = skills_mod

        pkg_prefix = skills_pkg
        pkg_dir = skills_root
        for part in parts[:-1]:
            pkg_prefix = f"{pkg_prefix}.{part}"
            pkg_dir = _os.path.join(pkg_dir, part)
            if pkg_prefix not in _sys.modules:
                pm = _types.ModuleType(pkg_prefix)
                pm.__path__ = [pkg_dir]
                pm.__package__ = pkg_prefix
                _sys.modules[pkg_prefix] = pm

        fqmn = skills_pkg if not parts else f"{skills_pkg}.{'.'.join(parts)}"

        if fqmn in _sys.modules:
            mod = _sys.modules[fqmn]
        else:
            subloc = [mod_path] if is_package else None
            spec = _iu.spec_from_file_location(fqmn, py_path, submodule_search_locations=subloc)
            if spec is None or spec.loader is None:
                raise BlueprintError(f"Unable to locate module '{module_name}' at '{py_path}'")
            mod = _iu.module_from_spec(spec)
            _sys.modules[fqmn] = mod
            spec.loader.exec_module(mod)
        return getattr(mod, attr_name) if attr_name else mod
    except Exception as e:
        raise BlueprintError(f"Failed to import child-local module '{module_name}': {e}") from e


def _is_async_callable(obj: Any) -> bool:
    try:
        import inspect

        return inspect.iscoroutinefunction(obj)
    except Exception:
        return False


class _PassthroughAgent:
    async def run(self, x: Any, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - trivial
        return x


def _resolve_agent(agent_spec: str) -> Any:
    obj = _import_object(agent_spec)
    try:
        import inspect

        if inspect.isclass(obj):
            return obj()
        return obj
    except Exception:
        return obj


def _resolve_agent_entry(agent: Union[str, JSONObject]) -> Any:
    if isinstance(agent, str):
        return _resolve_agent(agent)
    if isinstance(agent, dict):
        skill_id = agent.get("id")
        params = agent.get("params", {})
        if skill_id:
            resolver = get_skill_resolver()
            entry = resolver.get(skill_id, scope=None)
            if entry is None:
                try:
                    obj = _import_object(skill_id)
                    if callable(obj):
                        return obj(**params)
                    return obj
                except Exception:
                    raise BlueprintError(f"Unknown skill id: {skill_id}")
            factory = entry.get("factory")
            try:
                if callable(factory):
                    return factory(**params)
                return factory
            except TypeError as e:
                raise BlueprintError(f"Failed to instantiate skill '{skill_id}': {e}") from e
        path = agent.get("path")
        if path:
            return _resolve_agent(path)
    raise BlueprintError("Invalid agent specification")


def _resolve_plugins(specs: List[Union[str, JSONObject]]) -> List[Tuple[Any, int]]:
    result: List[Tuple[Any, int]] = []
    for item in specs:
        try:
            if isinstance(item, str):
                obj = _import_object(item)
                result.append((obj, 0))
            elif isinstance(item, dict):
                path = item.get("path")
                prio = int(item.get("priority", 0))
                if path:
                    obj = _import_object(path)
                    result.append((obj, prio))
        except Exception:
            continue
    return result


def _resolve_validators(specs: List[str]) -> List[Any]:
    result: List[Any] = []
    for path in specs:
        try:
            result.append(_import_object(path))
        except Exception:
            continue
    return result


__all__ = [
    "_current_skills_base_dir",
    "_import_object",
    "_is_async_callable",
    "_PassthroughAgent",
    "_pop_skills_base_dir",
    "_push_skills_base_dir",
    "_resolve_agent",
    "_resolve_agent_entry",
    "_resolve_plugins",
    "_resolve_validators",
]
