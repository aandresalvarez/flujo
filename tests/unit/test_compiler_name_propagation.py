import textwrap

from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml


def test_compiler_propagates_pipeline_name() -> None:
    yaml_text = textwrap.dedent(
        """
        version: "0.1"
        name: "unit_test_pipeline_name"

        agents:
          mini:
            model: "openai:gpt-4o-mini"
            system_prompt: |
              You are a tester. Return JSON with field 'summary'.
            output_schema:
              type: object
              properties:
                summary:
                  type: string
              required: [summary]

        steps:
          - name: s1
            uses: agents.mini
            input: "Hello"
            processing:
              structured_output: openai_json
              schema:
                type: object
                properties:
                  summary: { type: string }
                required: [summary]
        """
    ).strip()

    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)

    # The DeclarativeBlueprintCompiler path should set the Pipeline.name attribute
    assert hasattr(pipeline, "name"), "Compiled pipeline should have a name attribute"
    assert pipeline.name == "unit_test_pipeline_name"


def test_compiler_trims_pipeline_name_whitespace() -> None:
    yaml_text = textwrap.dedent(
        """
        version: "0.1"
        name: "  unit_test_pipeline_name  "

        agents:
          mini:
            model: "openai:gpt-4o-mini"
            system_prompt: |
              You are a tester. Return JSON with field 'summary'.
            output_schema:
              type: object
              properties:
                summary:
                  type: string
              required: [summary]

        steps:
          - name: s1
            uses: agents.mini
            input: "Hello"
            processing:
              structured_output: openai_json
              schema:
                type: object
                properties:
                  summary: { type: string }
                required: [summary]
        """
    ).strip()

    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)

    assert hasattr(pipeline, "name"), "Compiled pipeline should have a name attribute"
    assert pipeline.name == "unit_test_pipeline_name"
