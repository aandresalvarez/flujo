import re
import json
import uuid
from typing import Any, Dict
from pydantic import BaseModel
from .serialization import robust_serialize

IF_BLOCK_REGEX = re.compile(r"\{\{#if\s*([^\}]+?)\s*\}\}(.*?)\{\{\/if\}\}", re.DOTALL)
EACH_BLOCK_REGEX = re.compile(r"\{\{#each\s*([^\}]+?)\s*\}\}(.*?)\{\{\/each\}\}", re.DOTALL)
PLACEHOLDER_REGEX = re.compile(r"\{\{\s*([^\}]+?)\s*\}\}")


class AdvancedPromptFormatter:
    """Format prompt templates with conditionals, loops and nested data."""

    def __init__(self, template: str) -> None:
        """Initialize the formatter with a template string.

        Parameters
        ----------
        template:
            Template string containing ``{{`` placeholders and optional
            ``#if`` and ``#each`` blocks.
        """
        self.template = template
        # Generate a unique escape marker for this formatter instance
        # to prevent collisions with user content
        self._escape_marker = f"__ESCAPED_TEMPLATE_{uuid.uuid4().hex[:8]}__"

    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Any:
        """Retrieve ``key`` from ``data`` using dotted attribute syntax."""

        value: Any = data
        for part in key.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = getattr(value, part, None)
            if value is None:
                return None
        return value

    def _serialize_value(self, value: Any) -> str:
        """Serialize ``value`` to JSON using :func:`robust_serialize`."""
        serialized = robust_serialize(value)
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
                inner_formatter = AdvancedPromptFormatter(block)
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
            allowed = {"join", "upper", "lower", "length", "tojson"}
        if lname not in allowed:
            raise ValueError(f"Unknown template filter: {name}")
        if lname == "upper":
            return str(value).upper()
        if lname == "lower":
            return str(value).lower()
        if lname == "length":
            try:
                return len(value)
            except Exception:
                return 0
        if lname == "tojson":
            try:
                serialized = robust_serialize(value)
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
        import os as _os

        if _os.getenv("PYTEST_CURRENT_TEST"):
            _cached_ok = False
        else:
            _cached_ok = _CACHED_FILTERS is not None
    except Exception:
        _cached_ok = _CACHED_FILTERS is not None
    if _cached_ok:
        return _CACHED_FILTERS  # type: ignore[return-value]

    default = {"join", "upper", "lower", "length", "tojson"}
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

    def format(self, **kwargs: Any) -> str:
        """Render the template with the provided keyword arguments."""

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
                # Render the block with access to the current item via ``this``
                # without pre-inserting the serialized value. This prevents any
                # template syntax contained within ``item`` from being
                # interpreted a second time when the inner formatter runs.
                inner_formatter = AdvancedPromptFormatter(block)
                rendered = inner_formatter.format(**kwargs, this=item)
                # Escape any ``{{`` that appear in the rendered result so they
                # survive the outer placeholder pass unchanged.
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
            # Support simple "a or b or c" fallback semantics within placeholders.
            # Evaluate each candidate left-to-right; use the first truthy value.
            # - Identifiers are resolved via dotted lookup in kwargs.
            # - Single/double-quoted strings are treated as literals.
            # - Unknown/None/empty values are skipped.
            candidates = (
                [s.strip() for s in re.split(r"\s+or\s+", expr)] if " or " in expr else [expr]
            )
            chosen: Any = None
            for subexpr in candidates:
                # Quoted string literal
                if (len(subexpr) >= 2) and (
                    (subexpr[0] == subexpr[-1] == '"') or (subexpr[0] == subexpr[-1] == "'")
                ):
                    literal = subexpr[1:-1]
                    if literal:
                        chosen = literal
                        break
                    else:
                        continue
                # Variable/path lookup
                v = self._get_nested_value({**kwargs, **{"this": kwargs.get("this")}}, subexpr)
                # Treat empty string as falsy for fallback semantics
                if v is not None and (str(v) != ""):
                    chosen = v
                    break
            return chosen

        def placeholder_replacer(match: re.Match[str]) -> str:
            raw = match.group(1).strip()

            base_expr, filters = _split_filters(raw)
            value = _evaluate_with_fallback(base_expr)

            # Apply filters in order
            for flt in filters:
                value = _apply_filter(value, flt)

            # Serialize and escape chosen value (or empty string if none)
            serialized_value = self._serialize(value)
            return self._escape_template_syntax(serialized_value)

        processed = PLACEHOLDER_REGEX.sub(placeholder_replacer, processed)
        # Restore escaped template syntax
        processed = self._unescape_template_syntax(processed)
        return processed


def format_prompt(template: str, **kwargs: Any) -> str:
    """Convenience wrapper around :class:`AdvancedPromptFormatter`.

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

    formatter = AdvancedPromptFormatter(template)
    return formatter.format(**kwargs)
