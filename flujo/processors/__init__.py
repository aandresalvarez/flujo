from __future__ import annotations

from typing import Any, Awaitable, Callable, List, Optional, Protocol
import dataclasses
import orjson
import re
from pydantic import BaseModel as PydanticBaseModel, Field, ConfigDict
from typing import ClassVar


class Processor(Protocol):
    """Generic processor protocol."""

    name: str

    async def process(self, data: Any, context: Optional[PydanticBaseModel]) -> Any: ...


class AgentProcessors(PydanticBaseModel):
    """Container for prompt and output processors."""

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    prompt_processors: List[Any] = Field(default_factory=list)
    output_processors: List[Any] = Field(default_factory=list)


class AddContextVariables:
    """Prepends selected variables from the pipeline context to the prompt."""

    name = "AddContextVariables"

    def __init__(self, *, vars: List[str]):
        self.vars = vars

    async def process(self, data: Any, context: Optional[PydanticBaseModel]) -> Any:
        if not isinstance(data, str) or context is None:
            return data
        parts = []
        for var in self.vars:
            value = getattr(context, var, None)
            if value is not None:
                parts.append(f"{var}: {value}")
        if not parts:
            return data
        header = "--- CONTEXT ---\n" + "\n".join(parts) + "\n---\n"
        return header + data


class StripMarkdownFences:
    """Extracts content from a markdown code block."""

    name = "StripMarkdownFences"

    def __init__(self, *, language: str):
        self.language = language
        self._pattern = re.compile(rf"```{re.escape(language)}\s*(.*?)\s*```", re.DOTALL)

    async def process(self, data: Any, context: Optional[PydanticBaseModel]) -> Any:
        if not isinstance(data, str):
            return data
        match = self._pattern.search(data)
        if match:
            return match.group(1).strip()
        return data


class EnforceJsonResponse:
    """Ensures the output is valid JSON, optionally using a fixer callable."""

    name = "EnforceJsonResponse"

    def __init__(self, fixer: Optional[Callable[[str], Awaitable[str]]] = None):
        self.fixer = fixer

    async def process(self, data: Any, context: Optional[PydanticBaseModel]) -> Any:
        if not isinstance(data, str):
            return data
        try:
            orjson.loads(data)
            return data
        except Exception:
            if self.fixer is not None:
                fixed = await self.fixer(data)
                orjson.loads(fixed)  # may raise
                return fixed
            raise ValueError("Invalid JSON output")


__all__ = [
    "Processor",
    "AgentProcessors",
    "AddContextVariables",
    "StripMarkdownFences",
    "EnforceJsonResponse",
]
