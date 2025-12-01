from __future__ import annotations
from flujo.type_definitions.common import JSONObject

from typing import Any


def test_compiler_passes_model_settings(monkeypatch) -> None:
    """Ensure DeclarativeBlueprintCompiler forwards model_settings to make_agent_async."""
    from flujo.domain.blueprint.compiler import DeclarativeBlueprintCompiler
    from flujo.domain.blueprint.loader import BlueprintPipelineModel

    captured: JSONObject = {}

    def _fake_make_agent_async(*, model: str, system_prompt: str, output_type: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        captured["model"] = model
        captured["system_prompt"] = system_prompt
        captured["output_type"] = output_type
        captured["kwargs"] = kwargs

        class _A:
            async def run(self, x: Any, **_: Any) -> Any:
                return x

        return _A()

    monkeypatch.setattr(
        "flujo.domain.blueprint.compiler.make_agent_async",
        _fake_make_agent_async,
        raising=True,
    )

    bp = BlueprintPipelineModel.model_validate(
        {
            "version": "2.0",
            "agents": {
                "architect_agent": {
                    "model": "openai:gpt-5",
                    "system_prompt": "test",
                    "output_schema": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                        "required": ["x"],
                    },
                    "model_settings": {
                        "reasoning": {"effort": "high"},
                        "text": {"verbosity": "low"},
                    },
                    "timeout": 123,
                    "max_retries": 2,
                }
            },
            "steps": [{"kind": "step", "name": "S", "uses": "agents.architect_agent"}],
        }
    )

    comp = DeclarativeBlueprintCompiler(bp)
    pipe = comp.compile_to_pipeline()
    assert pipe is not None

    assert captured["model"] == "openai:gpt-5"
    assert isinstance(captured.get("kwargs"), dict)
    ms = captured["kwargs"].get("model_settings", {})
    assert ms.get("reasoning", {}).get("effort") == "high"
    assert ms.get("text", {}).get("verbosity") == "low"
    assert captured["kwargs"].get("timeout") == 123
    assert captured["kwargs"].get("max_retries") == 2
