from __future__ import annotations

import asyncio


def test_repair_yaml_ruamel_fixes_inline_steps_list() -> None:
    # Import builtins module to ensure registry side-effects ran
    import flujo.builtins as builtins

    INVALID = "version: '0.1'\nsteps: ["

    # Run the async repair helper
    result = asyncio.run(builtins.repair_yaml_ruamel(INVALID))

    assert isinstance(result, dict)
    text = result.get("generated_yaml") or result.get("yaml_text") or ""
    assert isinstance(text, str)
    # Heuristic must close the inline list to an empty list
    assert "steps: []" in text
    # Ensure version is present
    assert "version:" in text


def test_repair_yaml_ruamel_is_idempotent_on_valid_yaml() -> None:
    import flujo.builtins as builtins

    VALID = 'version: "0.1"\nname: pipeline\nsteps: []\n'
    result = asyncio.run(builtins.repair_yaml_ruamel(VALID))
    text = result.get("generated_yaml") or result.get("yaml_text") or ""
    assert isinstance(text, str)
    assert "steps: []" in text
    assert "version:" in text
