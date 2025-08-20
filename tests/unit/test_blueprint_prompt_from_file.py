from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import pytest


def _make_yaml(prompt_ref: str) -> str:
    return f"""
version: "0.1"
agents:
  a1:
    model: "openai:gpt-4o"
    system_prompt:
      from_file: {prompt_ref}
    output_schema: {{ type: "object" }}
steps:
  - kind: step
    name: S1
    uses: agents.a1
"""


def test_system_prompt_from_file_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_dir = tmp_path
    prompts_dir = base_dir / "prompts"
    prompts_dir.mkdir()
    prompt_file = prompts_dir / "a1.md"
    prompt_text = "Hello from prompt file!\nSecond line."
    prompt_file.write_text(prompt_text)

    yaml_text = _make_yaml("./prompts/a1.md")

    captured: List[Tuple[str, str]] = []

    # Patch compiler's make_agent_async to capture the system_prompt argument
    import flujo.domain.blueprint.compiler as compiler_mod

    def fake_make_agent_async(model: str, system_prompt: str, *args: Any, **kwargs: Any) -> Any:
        captured.append((model, system_prompt))

        async def _agent(x: Any, *a: Any, **k: Any) -> Any:
            return x

        return _agent

    monkeypatch.setattr(compiler_mod, "make_agent_async", fake_make_agent_async, raising=True)

    # Load via public loader with base_dir
    from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml

    _ = load_pipeline_blueprint_from_yaml(yaml_text, base_dir=str(base_dir))

    assert captured, "make_agent_async should be called for declarative agent"
    # The captured system_prompt must equal the file content
    assert captured[0][1] == prompt_text


def test_system_prompt_from_file_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_dir = tmp_path

    # Build model and compile directly to observe ConfigurationError type
    from flujo.domain.blueprint.loader import BlueprintPipelineModel
    from flujo.domain.blueprint.compiler import DeclarativeBlueprintCompiler
    from flujo.exceptions import ConfigurationError

    bp = BlueprintPipelineModel.model_validate(
        {
            "version": "0.1",
            "agents": {
                "a1": {
                    "model": "openai:gpt-4o",
                    "system_prompt": {"from_file": "./prompts/missing.md"},
                    "output_schema": {"type": "object"},
                }
            },
            "steps": [{"kind": "step", "name": "S1", "uses": "agents.a1"}],
        }
    )
    comp = DeclarativeBlueprintCompiler(bp, base_dir=str(base_dir))
    with pytest.raises(ConfigurationError):
        comp.compile_to_pipeline()


def test_system_prompt_from_file_sandbox(tmp_path: Path) -> None:
    base_dir = tmp_path / "proj"
    base_dir.mkdir()
    # Create a file outside project dir
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")

    from flujo.domain.blueprint.loader import BlueprintPipelineModel
    from flujo.domain.blueprint.compiler import DeclarativeBlueprintCompiler
    from flujo.exceptions import ConfigurationError

    bp = BlueprintPipelineModel.model_validate(
        {
            "version": "0.1",
            "agents": {
                "a1": {
                    "model": "openai:gpt-4o",
                    "system_prompt": {"from_file": "../secret.txt"},
                    "output_schema": {"type": "object"},
                }
            },
            "steps": [{"kind": "step", "name": "S1", "uses": "agents.a1"}],
        }
    )
    comp = DeclarativeBlueprintCompiler(bp, base_dir=str(base_dir))
    with pytest.raises(ConfigurationError):
        comp.compile_to_pipeline()


def test_system_prompt_plain_string_backward_compat(monkeypatch: pytest.MonkeyPatch) -> None:
    from flujo.domain.blueprint.loader import BlueprintPipelineModel
    from flujo.domain.blueprint.compiler import DeclarativeBlueprintCompiler

    bp = BlueprintPipelineModel.model_validate(
        {
            "version": "0.1",
            "agents": {
                "a1": {
                    "model": "openai:gpt-4o",
                    "system_prompt": "inline prompt",
                    "output_schema": {"type": "object"},
                }
            },
            "steps": [{"kind": "step", "name": "S1", "uses": "agents.a1"}],
        }
    )

    captured: list[str] = []
    import flujo.domain.blueprint.compiler as compiler_mod

    def fake_make_agent_async(model: str, system_prompt: str, *args: Any, **kwargs: Any) -> Any:
        captured.append(system_prompt)

        async def _agent(x: Any, *a: Any, **k: Any) -> Any:
            return x

        return _agent

    monkeypatch.setattr(compiler_mod, "make_agent_async", fake_make_agent_async, raising=True)

    comp = DeclarativeBlueprintCompiler(bp, base_dir=None)
    _ = comp.compile_to_pipeline()

    assert captured and captured[0] == "inline prompt"
