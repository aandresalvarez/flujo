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
        return json.dumps(serialized)

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

        def placeholder_replacer(match: re.Match[str]) -> str:
            key = match.group(1).strip()
            value = self._get_nested_value({**kwargs, **{"this": kwargs.get("this")}}, key)
            # Escape any template syntax that may appear inside user-provided
            # values so that it is rendered literally and not interpreted in a
            # subsequent regex pass.
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
