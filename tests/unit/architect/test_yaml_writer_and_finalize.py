from __future__ import annotations

import pytest

from flujo.architect.builder import _generate_yaml_from_tool_selections, _emit_minimal_yaml
from flujo.architect.context import ArchitectContext


@pytest.mark.asyncio
async def test_yaml_writer_disabled_builds_from_selections(monkeypatch: pytest.MonkeyPatch) -> None:
    # Disable YAML writer to exercise fallback assembly path
    monkeypatch.setenv("FLUJO_ARCHITECT_AGENTIC_YAMLWRITER", "0")

    ctx = ArchitectContext(
        user_goal="Search web and save",
        tool_selections=[
            {
                "step_name": "WebSearch",
                "chosen_agent_id": "flujo.builtins.web_search",
                "agent_params": {"query": "test"},
            },
            {
                "step_name": "SaveToFile",
                "chosen_agent_id": "flujo.builtins.fs_write_file",
                "agent_params": {"path": "out.txt"},
            },
        ],
    )

    out = await _generate_yaml_from_tool_selections(None, context=ctx)
    text = out.get("yaml_text") or ""
    assert "version:" in text and "steps:" in text
    assert "flujo.builtins.web_search" in text
    assert "flujo.builtins.fs_write_file" in text


@pytest.mark.asyncio
async def test_emit_minimal_yaml_produces_valid_scaffold() -> None:
    out = await _emit_minimal_yaml("My Pipeline")
    assert isinstance(out, dict)
    text = out.get("generated_yaml") or ""
    assert "version:" in text and "steps:" in text
