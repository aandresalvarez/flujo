from __future__ import annotations

from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml


def test_compiler_compiles_agents_before_steps() -> None:
    yaml_text = (
        'version: "0.1"\n'
        "agents:\n"
        "  decomposer:\n"
        '    model: "openai:gpt-4o-mini"\n'
        '    system_prompt: "You are a helper."\n'
        "    output_schema:\n"
        "      type: object\n"
        "      properties: { x: { type: string } }\n"
        "      required: [x]\n"
        "steps:\n"
        "  - kind: step\n"
        "    name: DecomposeGoal\n"
        "    uses: agents.decomposer\n"
    )

    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    assert pipeline.steps, "Pipeline should have at least one step"
    first = pipeline.steps[0]
    assert getattr(first, "agent", None) is not None, "Agent should be compiled and attached"
    # Ensure no import-string placeholder leaked
    assert not isinstance(first.agent, str)
