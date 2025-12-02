from __future__ import annotations

import fnmatch
import json
import os
from threading import RLock
from typing import Any, Iterable, Optional

from ..infra.telemetry import logfire

__all__ = ["BaseLinter", "_load_rule_overrides", "_override_severity", "logfire"]

from ..domain.pipeline_validation import ValidationFinding
from ..infra.config_manager import ConfigManager

# Conditional imports for TOML support (py311+ has tomllib, fallback to tomli)
try:
    import tomllib as toml_lib
except ImportError:
    try:
        import tomli as toml_lib  # type: ignore[no-redef]
    except ImportError:
        toml_lib = None  # type: ignore[assignment]

# --- Rule overrides (profile/file/env) for early skip and severity adjustment ---
_OVERRIDE_CACHE: Optional[dict[str, str]] = None
_OVERRIDE_CACHE_LOCK = RLock()


def _load_rule_overrides() -> dict[str, str]:
    """Load rule-id severity overrides from env/config in a thread-safe way."""
    global _OVERRIDE_CACHE
    if _OVERRIDE_CACHE is not None:
        return _OVERRIDE_CACHE
    with _OVERRIDE_CACHE_LOCK:
        if _OVERRIDE_CACHE is not None:
            return _OVERRIDE_CACHE
        mapping: dict[str, str] = {}
        # 1) Env JSON mapping (highest precedence for early-skip)
        try:
            env_json = os.getenv("FLUJO_RULES_JSON")
            if env_json:
                data = json.loads(env_json)
                if isinstance(data, dict):
                    mapping.update({str(k).upper(): str(v).lower() for k, v in data.items()})
        except Exception as e:
            logfire.debug(f"[validate] Invalid FLUJO_RULES_JSON: {e!r}")
        # 2) Env file path mapping
        try:
            rules_file = os.getenv("FLUJO_RULES_FILE")
            if rules_file and os.path.exists(rules_file):
                try:
                    if rules_file.endswith((".json", ".JSON")):
                        with open(rules_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data, dict):
                            mapping.update(
                                {str(k).upper(): str(v).lower() for k, v in data.items()}
                            )
                    elif rules_file.endswith((".toml", ".TOML")):
                        if toml_lib is None:
                            logfire.debug("[validate] TOML support not available (install tomli)")
                        else:
                            with open(rules_file, "rb") as f:
                                data = toml_lib.load(f)
                            # Expect { validation = { rules = {"V-T*"="off", ...} } }
                            try:
                                vm = data.get("validation", {}).get("rules", {})
                                if isinstance(vm, dict):
                                    mapping.update(
                                        {str(k).upper(): str(v).lower() for k, v in vm.items()}
                                    )
                            except Exception as e:
                                logfire.debug(
                                    f"[validate] TOML rules parse (validation.rules) failed: {e!r}"
                                )
                except Exception as e:
                    logfire.debug(
                        f"[validate] Failed reading FLUJO_RULES_FILE '{rules_file}': {e!r}"
                    )
        except Exception as e:
            logfire.debug(f"[validate] Error handling FLUJO_RULES_FILE: {e!r}")
        # 3) flujo.toml profile selected via FLUJO_RULES_PROFILE
        try:
            profile = os.getenv("FLUJO_RULES_PROFILE")
            if profile:
                cm = ConfigManager()
                cfg = cm.load_config()
                profiles = getattr(cfg, "validation", None)
                if profiles and getattr(profiles, "profiles", None):
                    raw = profiles.profiles.get(profile)
                    if isinstance(raw, dict):
                        mapping.update({str(k).upper(): str(v).lower() for k, v in raw.items()})
        except Exception as e:
            logfire.debug(f"[validate] Failed loading profile overrides: {e!r}")

        _OVERRIDE_CACHE = mapping
        return mapping


def _override_severity(rule_id: str, default: str) -> Optional[str]:
    """Return overridden severity ('error'/'warning') or None to indicate OFF."""
    mp = _load_rule_overrides()
    rid = str(rule_id).upper()
    if rid in mp:
        val = mp[rid]
    else:
        val = None
        for pat, sev in mp.items():
            try:
                if fnmatch.fnmatch(rid, pat):
                    val = sev
                    break
            except Exception:
                continue
    if not val:
        return default
    if val == "off":
        return None
    if val in {"warning", "error"}:
        return val
    return default


class BaseLinter:
    def analyze(self, pipeline: Any) -> Iterable[ValidationFinding]:  # pragma: no cover - interface
        return []
