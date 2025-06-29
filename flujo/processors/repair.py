from __future__ import annotations

import ast
import json
import re
from typing import Any, Final


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

        try:
            obj = ast.literal_eval(candidate)
            return self._canonical(obj)
        except Exception:
            pass

        raise ValueError("DeterministicRepairProcessor: unable to repair payload.")

    @staticmethod
    def _is_json(text: str) -> bool:
        try:
            json.loads(text)
            return True
        except Exception:
            return False

    @staticmethod
    def _canonical(data: Any) -> str:
        obj = data if not isinstance(data, str) else json.loads(data)
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def _balance(cls, text: str) -> str:
        opens, closes = text.count("{"), text.count("}")
        if opens > closes:
            text += "}" * (opens - closes)
        elif closes > opens:
            text = text.rstrip("}" * (closes - opens))
        diff = text.count("{") - text.count("}")
        if diff > 0:
            text += "}" * diff

        opens, closes = text.count("["), text.count("]")
        if opens > closes:
            text += "]" * (opens - closes)
        elif closes > opens:
            text = text.rstrip("]" * (closes - opens))
        diff = text.count("[") - text.count("]")
        if diff > 0:
            text += "]" * diff
        return text

    @classmethod
    def _repair_literals_and_quotes(cls, text: str) -> str:
        text = cls._RE_PY_LITERALS.sub(
            lambda m: {"None": "null", "True": "true", "False": "false"}[m.group(1)],
            text,
        )
        text = cls._RE_SINGLE_QUOTE.sub(lambda m: '"' + m.group(1) + '"', text)
        text = cls._RE_UNQUOTED_KEY.sub(r'\1"\2"\3', text)
        return text
