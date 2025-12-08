"""
JSON auto-repair utilities for agent outputs.

This module contains specialized logic for repairing malformed JSON outputs
from agents. It includes both deterministic repair strategies and LLM-based
repair functionality.

Extracted from flujo.infra.agents as part of FSD-005.3 to isolate specialized
repair logic from general agent functionality.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Final


from ..utils.serialization import safe_deserialize

# Import needed modules (avoiding circular imports)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .wrapper import AsyncAgentWrapper


# Import prompts from the prompts module
from ..prompts import REPAIR_SYS

MAX_LITERAL_EVAL_SIZE = 1_000_000


# Import from utils to avoid circular imports


class DeterministicRepairProcessor:
    """Tier-1 deterministic fixer for malformed JSON emitted by LLMs."""

    _RE_CODE_FENCE: Final = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I | re.M)
    _RE_LINE_COMMENT: Final = re.compile(r"(^|[^\S\r\n])//.*?$", re.M)
    _RE_HASH_COMMENT: Final = re.compile(r"(^|[^\S\r\n])#.*?$", re.M)
    _RE_BLOCK_COMMENT: Final = re.compile(r"/\*.*?\*/", re.S)
    _RE_TRAILING_COMMA: Final = re.compile(r",\s*([}\]])")
    _RE_SINGLE_QUOTE: Final = re.compile(r"(?<!\\)'([^'\\]*(?:\\.[^'\\]*)*)'")
    _RE_PY_LITERALS: Final = re.compile(r"\b(None|True|False)\b")
    _RE_UNQUOTED_KEY: Final = re.compile(r"([{\[,]\s*)([A-Za-z_][\w\-]*)(\s*:)")

    name: str = "DeterministicRepair"

    async def process(self, raw_output: str | bytes | Any) -> str:
        if isinstance(raw_output, bytes):
            raw_output = raw_output.decode()
        if not isinstance(raw_output, str):
            raise ValueError("DeterministicRepair expects a str or bytes payload.")

        if self._is_json(raw_output):
            return self._canonical(raw_output)

        candidate = raw_output.strip()

        try:
            obj, _ = json.JSONDecoder().raw_decode(candidate)
            return self._canonical(obj)
        except json.JSONDecodeError:
            pass

        candidate = self._RE_CODE_FENCE.sub("", candidate).strip()
        if self._is_json(candidate):
            return self._canonical(candidate)

        candidate = self._RE_BLOCK_COMMENT.sub("", candidate)
        candidate = self._RE_LINE_COMMENT.sub(r"\1", candidate)
        candidate = self._RE_HASH_COMMENT.sub(r"\1", candidate)
        if self._is_json(candidate):
            return self._canonical(candidate)

        candidate = self._RE_TRAILING_COMMA.sub(r"\1", candidate)
        if self._is_json(candidate):
            return self._canonical(candidate)

        candidate = self._balance(candidate)
        if self._is_json(candidate):
            return self._canonical(candidate)

        candidate = self._repair_literals_and_quotes(candidate)
        if self._is_json(candidate):
            return self._canonical(candidate)

        if len(candidate) > MAX_LITERAL_EVAL_SIZE:
            raise ValueError("Input too large for safe literal evaluation.")

        try:
            obj = ast.literal_eval(candidate)
            return self._canonical(obj)
        except Exception as e:
            import logging

            logging.warning(f"DeterministicRepairProcessor: ast.literal_eval failed: {e}")
            # pass

        raise ValueError("DeterministicRepairProcessor: unable to repair payload.")

    @staticmethod
    def _is_json(text: str) -> bool:
        """Return ``True`` if ``text`` is valid JSON."""
        try:
            json.loads(text)
            return True
        except Exception as e:
            import logging

            logging.warning(f"DeterministicRepairProcessor: _is_json failed: {e}")
            return False

    @staticmethod
    def _canonical(data: Any) -> str:
        """Serialize ``data`` to canonical JSON string form."""
        obj = data if not isinstance(data, str) else safe_deserialize(json.loads(data))

        def _serialize(obj: Any) -> Any:
            try:
                from pydantic import BaseModel as _BM

                if isinstance(obj, _BM):
                    return obj.model_dump(mode="json")
            except Exception:
                pass
            try:
                import dataclasses as _dc

                if _dc.is_dataclass(obj) and not isinstance(obj, type):
                    return _dc.asdict(obj)
            except Exception:
                pass
            return obj

        serialized = _serialize(obj)
        return json.dumps(serialized, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def _balance(cls, text: str) -> str:
        """Balance braces and brackets by only adjusting the tail."""
        curly_open = curly_close = square_open = square_close = 0
        in_str = False
        escape = False
        quote = ""
        for ch in text:
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_str = False
                continue
            elif ch in ('"', "'"):
                in_str = True
                quote = ch
            elif ch == "{":
                curly_open += 1
            elif ch == "}":
                curly_close += 1
            elif ch == "[":
                square_open += 1
            elif ch == "]":
                square_close += 1

        diff = curly_open - curly_close
        if diff > 0:
            text += "}" * diff
        elif diff < 0:
            remove = min(-diff, len(text) - len(text.rstrip("}")))
            if remove:
                text = text[:-remove]

        diff = square_open - square_close
        if diff > 0:
            text += "]" * diff
        elif diff < 0:
            remove = min(-diff, len(text) - len(text.rstrip("]")))
            if remove:
                text = text[:-remove]
        return text

    @classmethod
    def _repair_literals_and_quotes(cls, text: str) -> str:
        """Fix common JSON issues like Python literals and single quotes."""
        text = cls._RE_PY_LITERALS.sub(
            lambda m: {"None": "null", "True": "true", "False": "false"}[m.group(1)],
            text,
        )
        text = cls._RE_SINGLE_QUOTE.sub(lambda m: '"' + m.group(1) + '"', text)
        text = cls._RE_UNQUOTED_KEY.sub(r'\1"\2"\3', text)
        return text


def make_repair_agent(model: str | None = None) -> "AsyncAgentWrapper[Any, str]":
    """Create the internal JSON repair agent."""
    from ..infra.settings import settings

    # Import make_agent_async via delegation to avoid circular imports
    from .wrapper import make_agent_async

    model_name = model or settings.default_repair_model
    return make_agent_async(model_name, REPAIR_SYS, str, auto_repair=False)


# Global repair agent instance for lazy initialization
_repair_agent: "AsyncAgentWrapper[Any, str]" | None = None


def get_repair_agent() -> "AsyncAgentWrapper[Any, str]":
    """Lazily create the internal repair agent."""
    global _repair_agent
    if _repair_agent is None:
        _repair_agent = make_repair_agent()
    return _repair_agent
