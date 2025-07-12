from __future__ import annotations

import re
import json
from typing import Any, Optional, List

from ..utils.serialization import safe_deserialize

from ..domain.models import BaseModel


class AddContextVariables:
    """Prepend context variables to a prompt."""

    def __init__(self, vars: List[str]):
        self.vars = vars
        self.name = "AddContextVariables"

    async def process(self, data: Any, context: Optional[BaseModel] = None) -> Any:
        if context is None:
            return data
        header_lines = ["--- CONTEXT ---"]
        for var in self.vars:
            value = getattr(context, var, None)
            header_lines.append(f"{var}: {value}")
        header_lines.append("---")
        prefix = "\n".join(header_lines)
        return f"{prefix}\n{data}"


class StripMarkdownFences:
    """Extract content from a fenced code block."""

    def __init__(self, language: str):
        self.language = language
        self.name = "StripMarkdownFences"
        self.pattern = re.compile(rf"```{re.escape(language)}\n(.*?)\n```", re.DOTALL)

    async def process(self, data: Any, context: Optional[BaseModel] = None) -> Any:
        if not isinstance(data, str):
            return data
        match = self.pattern.search(data)
        if match:
            return match.group(1).strip()
        return data


class EnforceJsonResponse:
    """Ensure the output is valid JSON."""

    def __init__(self) -> None:
        self.name = "EnforceJsonResponse"

    async def process(self, data: Any, context: Optional[BaseModel] = None) -> Any:
        if isinstance(data, (dict, list)):
            return data
        if not isinstance(data, str):
            return data
        try:
            return safe_deserialize(json.loads(data))
        except Exception:
            return data


class SerializePydantic:
    """Serialize any object with a ``model_dump`` method to a plain ``dict``."""

    def __init__(self) -> None:
        self.name = "SerializePydantic"

    async def process(self, data: Any, context: Optional[BaseModel] = None) -> Any:
        dump = getattr(data, "model_dump", None)
        if callable(dump):
            return dump()
        return data
