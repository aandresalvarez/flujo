from __future__ import annotations

from typing import Any

import pytest

from flujo.exceptions import ConfigurationError
from flujo.infra.skill_registry import SkillRegistry


def test_skill_registry_attaches_skill_id_to_tool() -> None:
    async def my_tool(x: int) -> int:
        return x + 1

    reg = SkillRegistry()
    reg.register("acme.my_tool", lambda **_: my_tool)
    entry = reg.get("acme.my_tool")
    assert entry is not None
    tool = entry["factory"]()
    assert getattr(tool, "__flujo_skill_id__", None) == "acme.my_tool"


@pytest.mark.asyncio
async def test_skill_registry_enforces_allowlist_at_call_time(monkeypatch: Any) -> None:
    monkeypatch.setenv("FLUJO_GOVERNANCE_TOOL_ALLOWLIST", "acme.allowed")

    async def allowed() -> str:
        return "ok"

    async def denied() -> str:
        return "no"

    reg = SkillRegistry()
    reg.register("acme.allowed", lambda **_: allowed)
    reg.register("acme.denied", lambda **_: denied)

    entry_ok = reg.get("acme.allowed")
    entry_no = reg.get("acme.denied")
    assert entry_ok is not None
    assert entry_no is not None

    tool_ok = entry_ok["factory"]()
    tool_no = entry_no["factory"]()

    assert await tool_ok() == "ok"
    with pytest.raises(ConfigurationError, match="tool_not_allowed:acme\\.denied"):
        await tool_no()
