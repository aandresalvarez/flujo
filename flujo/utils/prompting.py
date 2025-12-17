from __future__ import annotations
import re
import json
import uuid
from typing import Any
from pydantic import BaseModel
from .serialization import _robust_serialize_internal
from flujo.type_definitions.common import JSONObject

IF_BLOCK_REGEX = re.compile(r"\{\{#if\s*([^\}]+?)\s*\}\}(.*?)\{\{\/if\}\}", re.DOTALL)
EACH_BLOCK_REGEX = re.compile(r"\{\{#each\s*([^\}]+?)\s*\}\}(.*?)\{\{\/each\}\}", re.DOTALL)
PLACEHOLDER_REGEX = re.compile(r"\{\{\s*([^\}]+?)\s*\}\}")


class AdvancedPromptFormatter:
    """Format prompt templates with conditionals, loops and nested data."""

    def __init__(
        self, template: str, *, strict: bool = False, log_resolution: bool = False
    ) -> None:
        """Initialize the formatter with a template string.

        Parameters
        ----------
        template:
            Template string containing ``{{`` placeholders and optional
            ``#if`` and ``#each`` blocks.
        strict:
            If True, raise TemplateResolutionError on undefined variables.
            If False, return None (resolves to empty string).
        log_resolution:
            If True, log template resolution process for debugging.
        """
        self.template = template
        self._strict = strict
        self._log_resolution = log_resolution
        # Generate a unique escape marker for this formatter instance
        # to prevent collisions with user content
        self._escape_marker = f"__ESCAPED_TEMPLATE_{uuid.uuid4().hex[:8]}__"

    def _get_nested_value(self, data: JSONObject, key: str) -> Any:
        """Retrieve ``key`` from ``data`` using dotted attribute syntax.

        In strict mode, raises TemplateResolutionError if key is undefined.
        """
        from ..exceptions import TemplateResolutionError

        value: Any = data
        path_parts: list[str] = []

        for part in key.split("."):
            path_parts.append(part)
            if isinstance(value, dict):
                if part not in value:
                    # Undefined variable
                    if self._strict:
                        available = list(data.keys()) if isinstance(data, dict) else []
                        raise TemplateResolutionError(
                            f"Undefined template variable: '{key}' "
                            f"(failed at '{'.'.join(path_parts)}'). "
                            f"Available variables: {available}"
                        )
                    return None
                value = value.get(part)
            else:
                if not hasattr(value, part):
                    # Undefined attribute
                    if self._strict:
                        available = list(data.keys()) if isinstance(data, dict) else []
                        raise TemplateResolutionError(
                            f"Undefined template variable: '{key}' "
                            f"(failed at '{'.'.join(path_parts)}'). "
                            f"Available variables: {available}"
                        )
                    return None
                value = getattr(value, part, None)
            if value is None:
                return None
        return value

    def _serialize_value(self, value: Any) -> str:
        """Serialize ``value`` to JSON using internal robust serializer."""
        serialized = _robust_serialize_internal(value)
        # If robust_serialize returns a string, it's already serialized
        if isinstance(serialized, str):
            return serialized
        # Otherwise, serialize to JSON with fallback handling
        try:
            return json.dumps(serialized)
        except (TypeError, ValueError):
            # If JSON serialization fails, return a string representation
            return str(serialized)

    def _serialize(self, value: Any) -> str:
        """Serialize ``value`` for interpolation into a template."""

        if value is None:
            return ""
        if isinstance(value, BaseModel):
            # Use robust serialization instead of model_dump_json to avoid failures on unknown types
            return self._serialize_value(value)
        if isinstance(value, (dict, list)):
            # Use enhanced serialization instead of orjson
            return self._serialize_value(value)
        return str(value)

    def _escape_template_syntax(self, text: str) -> str:
        """Escape template syntax in user-provided content.

        This method safely escapes {{ in user content without affecting
        literal occurrences of the escape marker in user data.
        """
        # Replace {{ with our unique escape marker
        return text.replace("{{", self._escape_marker)

    def _unescape_template_syntax(self, text: str) -> str:
        """Restore escaped template syntax.

        This method converts our unique escape marker back to {{.
        """
        return text.replace(self._escape_marker, "{{")

    def format(self, **kwargs: Any) -> str:
        """Render the template with the provided keyword arguments."""

        # Log template resolution if enabled
        if self._log_resolution:
            try:
                from ..infra import telemetry

                telemetry.logfire.debug(f"[TEMPLATE] Rendering: {self.template[:100]}...")
                telemetry.logfire.debug(f"[TEMPLATE] Available variables: {list(kwargs.keys())}")
            except Exception:
                pass

        # First, escape literal \{{ in the template
        processed = self.template.replace(r"\{{", self._escape_marker)

        def if_replacer(match: re.Match[str]) -> str:
            key, content = match.groups()
            value = self._get_nested_value(kwargs, key.strip())
            return content if value else ""

        processed = IF_BLOCK_REGEX.sub(if_replacer, processed)

        def each_replacer(match: re.Match[str]) -> str:
            key, block = match.groups()
            items = self._get_nested_value(kwargs, key.strip())
            if not isinstance(items, list):
                return ""
            parts = []
            for item in items:
                inner_formatter = AdvancedPromptFormatter(
                    block,
                    strict=self._strict,
                    log_resolution=self._log_resolution,
                )
                rendered = inner_formatter.format(**kwargs, this=item)
                rendered = self._escape_template_syntax(rendered)
                parts.append(rendered)
            return "".join(parts)

        processed = EACH_BLOCK_REGEX.sub(each_replacer, processed)

        def _split_filters(expr: str) -> tuple[str, list[str]]:
            parts: list[str] = []
            buf: list[str] = []
            in_single = False
            in_double = False
            paren = 0
            for ch in expr:
                if ch == "'" and not in_double:
                    in_single = not in_single
                    buf.append(ch)
                    continue
                if ch == '"' and not in_single:
                    in_double = not in_double
                    buf.append(ch)
                    continue
                if ch == "(" and not in_single and not in_double:
                    paren += 1
                    buf.append(ch)
                    continue
                if ch == ")" and not in_single and not in_double and paren > 0:
                    paren -= 1
                    buf.append(ch)
                    continue
                if ch == "|" and not in_single and not in_double and paren == 0:
                    parts.append("".join(buf).strip())
                    buf = []
                else:
                    buf.append(ch)
            if buf:
                parts.append("".join(buf).strip())
            if not parts:
                return expr, []
            base = parts[0]
            filters = parts[1:]
            return base, filters

        def _apply_filter(value: Any, flt: str) -> Any:
            return AdvancedPromptFormatter._apply_filter_static(value, flt)

        def _evaluate_with_fallback(expr: str) -> Any:
            candidates = (
                [s.strip() for s in re.split(r"\s+or\s+", expr)] if " or " in expr else [expr]
            )
            chosen: Any = None
            for subexpr in candidates:
                if (len(subexpr) >= 2) and (
                    (subexpr[0] == subexpr[-1] == '"') or (subexpr[0] == subexpr[-1] == "'")
                ):
                    literal = subexpr[1:-1]
                    if literal:
                        chosen = literal
                        break
                    else:
                        continue
                v = self._get_nested_value({**kwargs, **{"this": kwargs.get("this")}}, subexpr)
                if v is not None and (str(v) != ""):
                    chosen = v
                    break
            return chosen

        def placeholder_replacer(match: re.Match[str]) -> str:
            raw = match.group(1).strip()
            base_expr, filters = _split_filters(raw)
            value = _evaluate_with_fallback(base_expr)
            for flt in filters:
                value = _apply_filter(value, flt)
            serialized_value = self._serialize(value)
            return self._escape_template_syntax(serialized_value)

        processed = PLACEHOLDER_REGEX.sub(placeholder_replacer, processed)
        processed = self._unescape_template_syntax(processed)

        # Log resolved result and warn if empty when template had content
        if self._log_resolution or (processed == "" and self.template.strip() != ""):
            try:
                from ..infra import telemetry

                if self._log_resolution:
                    telemetry.logfire.debug(f"[TEMPLATE] Resolved to: {processed[:100]}...")
                if processed == "" and self.template.strip() != "" and "{{" in self.template:
                    telemetry.logfire.warning(
                        f"[TEMPLATE] Template resolved to empty string! "
                        f"Original template: '{self.template[:100]}...' "
                        f"Available variables: {list(kwargs.keys())}"
                    )
            except Exception:
                pass

        return processed

    @staticmethod
    def _apply_filter_static(value: Any, flt: str) -> Any:
        name = flt
        arg: str | None = None
        if "(" in flt and flt.endswith(")"):
            name = flt[: flt.index("(")].strip()
            arg_str = flt[flt.index("(") + 1 : -1].strip()
            if arg_str:
                if (len(arg_str) >= 2) and (
                    (arg_str[0] == arg_str[-1] == '"') or (arg_str[0] == arg_str[-1] == "'")
                ):
                    arg = arg_str[1:-1]
                else:
                    arg = arg_str

        lname = name.lower()
        try:
            allowed = _get_enabled_filters()
        except Exception:
            allowed = {"join", "upper", "lower", "length", "tojson", "default"}
        if lname not in allowed:
            raise ValueError(f"Unknown template filter: {name}")
        if lname == "upper":
            return str(value).upper()
        if lname == "lower":
            return str(value).lower()
        if lname == "default":
            return value if value not in (None, "", []) else (arg or "")
        if lname == "length":
            try:
                return len(value)
            except Exception:
                return 0
        if lname == "tojson":
            try:
                serialized = _robust_serialize_internal(value)
            except Exception:
                serialized = value
            try:
                return json.dumps(serialized)
            except Exception:
                return json.dumps(str(serialized))
        if lname == "join":
            delim = arg or ""
            if isinstance(value, (list, tuple)):
                try:
                    return delim.join(str(x) for x in value)
                except Exception:
                    return delim.join([str(value)])
            return str(value)
        raise ValueError(f"Unknown template filter: {name}")


_CACHED_FILTERS: set[str] | None = None


def _get_enabled_filters() -> set[str]:
    """Return the set of enabled template filters from configuration.

    Reads flujo.toml via ConfigManager settings.enabled_template_filters when available.
    Falls back to the default allow-list when not configured.
    """
    global _CACHED_FILTERS
    # In test/CI contexts, avoid cross-test stale cache that may hide unknown-filter warnings
    try:
        from ..infra.settings import get_settings

        test_mode = bool(get_settings().test_mode)
    except Exception:
        test_mode = False
    _cached_ok = _CACHED_FILTERS is not None and not test_mode
    if _cached_ok:
        return _CACHED_FILTERS  # type: ignore[return-value]

    default = {"join", "upper", "lower", "length", "tojson", "default"}
    try:
        from ..infra.config_manager import get_config_manager

        cfg = get_config_manager().load_config()
        settings = getattr(cfg, "settings", None)
        enabled = getattr(settings, "enabled_template_filters", None) if settings else None
        if isinstance(enabled, list) and all(isinstance(x, str) for x in enabled):
            import re

            valid = {s.lower() for s in enabled if re.fullmatch(r"[a-z_]+", s.lower())}
            _CACHED_FILTERS = valid or default
            return _CACHED_FILTERS
    except Exception:
        pass
    _CACHED_FILTERS = default
    return _CACHED_FILTERS


def format_prompt(template: str, **kwargs: Any) -> str:
    """Convenience wrapper around :class:`AdvancedPromptFormatter`.

    This helper respects the global template configuration from flujo.toml,
    including strict mode and resolution logging.

    Parameters
    ----------
    template:
        Template string to render.
    **kwargs:
        Values referenced inside the template.

    Returns
    -------
    str
        The rendered template.
    """
    # Load template configuration to honor strict mode and logging settings
    strict = False
    log_resolution = False
    try:
        from flujo.infra.config_manager import get_config_manager, TemplateConfig

        config_mgr = get_config_manager()
        config = config_mgr.load_config()
        template_config = config.template or TemplateConfig()
        strict = template_config.undefined_variables == "strict"
        log_resolution = template_config.log_resolution
    except Exception:
        # Fallback to defaults if config unavailable
        pass

    formatter = AdvancedPromptFormatter(template, strict=strict, log_resolution=log_resolution)
    return formatter.format(**kwargs)
