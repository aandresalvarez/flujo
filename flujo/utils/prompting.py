import re
import orjson
from typing import Any, Dict
from pydantic import BaseModel

IF_BLOCK_REGEX = re.compile(r"\{\{#if\s*([^\}]+?)\s*\}\}(.*?)\{\{\/if\}\}", re.DOTALL)
EACH_BLOCK_REGEX = re.compile(r"\{\{#each\s*([^\}]+?)\s*\}\}(.*?)\{\{\/each\}\}", re.DOTALL)
PLACEHOLDER_REGEX = re.compile(r"\{\{\s*([^\}]+?)\s*\}\}")


class AdvancedPromptFormatter:
    """Format prompt templates with conditionals, loops and nested data."""

    def __init__(self, template: str) -> None:
        self.template = template

    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Any:
        value: Any = data
        for part in key.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = getattr(value, part, None)
            if value is None:
                return None
        return value

    def _serialize(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, BaseModel):
            return value.model_dump_json()
        if isinstance(value, (dict, list)):
            return str(orjson.dumps(value).decode())
        return str(value)

    def format(self, **kwargs: Any) -> str:
        ESC_MARKER = "__ESCAPED_OPEN__"
        processed = self.template.replace(r"\{{", ESC_MARKER)

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
                inner = block.replace("{{ this }}", self._serialize(item))
                # allow nested placeholders referring to outer scope as well
                inner_formatter = AdvancedPromptFormatter(inner)
                parts.append(inner_formatter.format(**kwargs, this=item))
            return "".join(parts)

        processed = EACH_BLOCK_REGEX.sub(each_replacer, processed)

        def placeholder_replacer(match: re.Match[str]) -> str:
            key = match.group(1).strip()
            value = self._get_nested_value({**kwargs, **{"this": kwargs.get("this")}}, key)
            return self._serialize(value)

        processed = PLACEHOLDER_REGEX.sub(placeholder_replacer, processed)
        processed = processed.replace(ESC_MARKER, "{{")
        return processed


def format_prompt(template: str, **kwargs: Any) -> str:
    formatter = AdvancedPromptFormatter(template)
    return formatter.format(**kwargs)
