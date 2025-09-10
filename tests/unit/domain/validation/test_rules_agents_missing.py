from __future__ import annotations

from flujo.domain.dsl import Pipeline, Step


def test_v_a6_unknown_agent_or_skill_emits_error() -> None:
    """V-A6: If a step.agent is a string import path that cannot be resolved, emit error.

    Build a simple pipeline with agent as an invalid import path.
    """

    s = Step.model_validate({"name": "s", "agent": "does.not.exist:fn"})
    report = Pipeline.from_step(s).validate_graph()
    errs = [e for e in report.errors if e.rule_id == "V-A6" and e.step_name == "s"]
    assert errs, f"Expected V-A6 error; got: {report.model_dump()}"


def test_v_a7_invalid_coercions_warns(tmp_path) -> None:
    """V-A7: Declarative agents with invalid max_retries/timeout produce warnings."""
    yml = (
        'version: "0.1"\n'
        "agents:\n"
        "  bad:\n"
        '    model: "openai:gpt-4o"\n'
        '    system_prompt: "x"\n'
        "    output_schema: { type: object }\n"
        '    max_retries: "three"\n'
        '    timeout: "ten seconds"\n'
        "steps:\n"
        "  - name: s\n"
        "    uses: agents.bad\n"
    )
    p = (tmp_path / "p.yaml").resolve()
    p.write_text(yml)
    from flujo.cli.helpers import validate_pipeline_file

    report = validate_pipeline_file(str(p), include_imports=False)
    warns = [w for w in report.warnings if w.rule_id == "V-A7" and w.step_name == "s"]
    assert warns, report.model_dump()


def test_v_a8_structured_output_non_json_mode_todo() -> None:
    """FSD: V-A8 — Structured output requested but provider in non-JSON mode; warn."""
    import tempfile
    from pathlib import Path

    yml = (
        'version: "0.1"\n'
        "agents:\n"
        "  a:\n"
        '    model: "openai:gpt-4o"\n'
        '    system_prompt: "x"\n'
        "    output_schema: { type: object, properties: { ok: { type: boolean } } }\n"
        "steps:\n"
        "  - name: s\n"
        "    uses: agents.a\n"
        "    processing:\n"
        '      structured_output: "off"\n'
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "p.yaml"
        p.write_text(yml)
        from flujo.cli.helpers import validate_pipeline_file

        report = validate_pipeline_file(p.as_posix(), include_imports=False)
        warns = [w for w in report.warnings if w.rule_id == "V-A8" and w.step_name == "s"]
        assert warns, report.model_dump()
