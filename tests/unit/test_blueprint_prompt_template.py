from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import pytest


def _make_yaml(path_ref: str) -> str:
    return f"""
version: "0.1"
agents:
  templated:
    model: "openai:gpt-4o-mini"
    system_prompt:
      from_file: {path_ref}
      variables:
        max_sentences: 5
        topic: "{{{{ context.topic }}}}"
        author: "{{{{ previous_step.author_name }}}}"
    output_schema: {{ type: "object" }}
steps:
  - kind: step
    name: S1
    uses: agents.templated
"""


def test_compiler_wires_templated_wrapper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_dir = tmp_path
    prompts_dir = base_dir / "prompts"
    prompts_dir.mkdir()
    prompt_file = prompts_dir / "tmpl.md"
    prompt_text = (
        "Summarize {{ topic }} in {{ max_sentences }} sentences by {{ author }}.\n"
        "User: {{ context.user }} Prev: {{ previous_step.value }}"
    )
    prompt_file.write_text(prompt_text)

    yaml_text = _make_yaml("./prompts/tmpl.md")

    captured: List[Tuple[str, str, dict[str, Any]]] = []

    # Patch compiler's make_templated_agent_async to capture arguments
    import flujo.domain.blueprint.compiler as compiler_mod

    def fake_make_templated_agent_async(
        *,
        model: str,
        template_string: str,
        variables_spec: Optional[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        captured.append((model, template_string, variables_spec or {}))

        async def _agent(x: Any, *a: Any, **k: Any) -> Any:  # pragma: no cover - placeholder
            return x

        return _agent

    monkeypatch.setattr(
        compiler_mod, "make_templated_agent_async", fake_make_templated_agent_async, raising=True
    )

    # Load via public loader with base_dir
    from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml

    _ = load_pipeline_blueprint_from_yaml(yaml_text, base_dir=str(base_dir))

    assert captured, "make_templated_agent_async should be called for templated prompt spec"
    model_name, template_string, variables = captured[0]
    assert template_string == prompt_text
    assert variables["max_sentences"] == 5
    assert variables["topic"] == "{{ context.topic }}"
    assert variables["author"] == "{{ previous_step.author_name }}"
