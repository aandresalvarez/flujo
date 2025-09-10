from __future__ import annotations

from pathlib import Path


def test_v_s2_response_format_vs_stringification(tmp_path: Path) -> None:
    """V-S2: Structured output then stringified by next step."""
    yml = (
        'version: "0.1"\n'
        "agents:\n"
        "  a:\n"
        '    model: "openai:gpt-4o"\n'
        '    system_prompt: "x"\n'
        "    output_schema: { type: object }\n"
        "steps:\n"
        "  - name: s1\n"
        "    uses: agents.a\n"
        "  - name: s2\n"
        '    agent: { id: "flujo.builtins.stringify" }\n'
        '    input: "{{ steps.s1.output }}"\n'
    )
    p = tmp_path / "p.yaml"
    p.write_text(yml)
    from flujo.cli.helpers import validate_pipeline_file

    report = validate_pipeline_file(str(p), include_imports=False)
    assert any(w.rule_id == "V-S2" and w.step_name == "s2" for w in report.warnings), (
        report.model_dump()
    )


def test_v_s3_type_string_awareness(tmp_path: Path) -> None:
    """V-S3: output_schema uses type=string; emit awareness warning."""
    yml = (
        'version: "0.1"\n'
        "agents:\n"
        "  a:\n"
        '    model: "openai:gpt-4o"\n'
        '    system_prompt: "x"\n'
        "    output_schema: { type: string }\n"
        "steps:\n"
        "  - name: s1\n"
        "    uses: agents.a\n"
    )
    p = tmp_path / "p.yaml"
    p.write_text(yml)
    from flujo.cli.helpers import validate_pipeline_file

    report = validate_pipeline_file(str(p), include_imports=False)
    assert any(w.rule_id == "V-S3" and w.step_name == "s1" for w in report.warnings), (
        report.model_dump()
    )
