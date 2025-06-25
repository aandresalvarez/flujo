from __future__ import annotations

import re
import orjson
from typing import Any, List, Optional, Protocol
from pydantic import BaseModel, Field


class Processor(Protocol):
    """Interface for prompt or output processors."""

    name: str

    async def process(self, data: Any, context: Optional[BaseModel]) -> Any: ...


class AgentProcessors(BaseModel):
    """Container for prompt and output processors."""

    prompt_processors: List[Any] = Field(default_factory=list)
    output_processors: List[Any] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class AddContextVariables:
    """Prepends selected context variables to the prompt string."""

    def __init__(self, *, vars: List[str]):
        self.vars = vars
        self.name = "AddContextVariables"

    async def process(self, data: Any, context: Optional[BaseModel]) -> Any:
        if context is None or not isinstance(data, str):
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
    """Extracts content from a fenced code block."""

    def __init__(self, *, language: str):
        self.language = language
        self.name = "StripMarkdownFences"
        self._pattern = re.compile(rf"```{re.escape(language)}\s*(.*?)\s*```", re.DOTALL)

    async def process(self, data: Any, context: Optional[BaseModel]) -> Any:
        if not isinstance(data, str):
            return data
        match = self._pattern.search(data)
        if match:
            return match.group(1).strip()
        return data.strip()


class EnforceJsonResponse:
    """Ensures the output is valid JSON, optionally using a fixer agent."""

    def __init__(self, fixer_agent: Any | None = None) -> None:
        self.fixer_agent = fixer_agent
        self.name = "EnforceJsonResponse"

    async def process(self, data: Any, context: Optional[BaseModel]) -> Any:
        text = data if isinstance(data, str) else str(data)
        try:
            orjson.loads(text)
            return data
        except Exception:
            if self.fixer_agent is not None:
                fixed = await self.fixer_agent.run(text)
                orjson.loads(fixed)
                return fixed
            raise


__all__ = [
    "Processor",
    "AgentProcessors",
    "AddContextVariables",
    "StripMarkdownFences",
    "EnforceJsonResponse",
]
