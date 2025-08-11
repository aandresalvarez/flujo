from __future__ import annotations

from typing import Any

try:
    from sqlvalidator import parse
except ImportError:
    # Fallback stub when sqlvalidator is not installed
    def parse(query: str):
        class DummyResult:
            def __init__(self, query: str):
                self.errors = ["invalid SQL"]

            def is_valid(self) -> bool:
                return False

        return DummyResult(query)


from ..domain.plugins import PluginOutcome


class SQLSyntaxValidator:
    """Validation plugin that checks SQL syntax."""

    async def validate(self, data: dict[str, Any]) -> PluginOutcome:
        query = data.get("output") or data.get("solution") or data.get("query") or ""
        try:
            result = parse(query)
            try:
                valid = result.is_valid()
                errors = "; ".join(result.errors)
            except Exception as e:
                valid = False
                errors = str(e)
            if valid:
                return PluginOutcome(success=True)
            return PluginOutcome(success=False, feedback=errors or "invalid SQL")
        except Exception as e:  # pragma: no cover - unexpected parse errors
            return PluginOutcome(success=False, feedback=str(e))
