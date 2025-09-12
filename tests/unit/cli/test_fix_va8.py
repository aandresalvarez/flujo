from __future__ import annotations

from pathlib import Path


def _write(tmp_path: Path, name: str, text: str) -> str:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p.as_posix()


def _count_rule(report, rid: str) -> int:
    errs = getattr(report, "errors", []) or []
    wrns = getattr(report, "warnings", []) or []
    return sum(1 for e in errs if getattr(e, "rule_id", "") == rid) + sum(
        1 for w in wrns if getattr(w, "rule_id", "") == rid
    )


def test_fix_va8_adds_structured_output(tmp_path: Path) -> None:
    yaml_text = (
        'version: "0.1"\n'
        "steps:\n"
        "  - kind: step\n"
        "    name: Example\n"
        "    agent: { id: 'flujo.builtins.stringify' }\n"
        "    processing:\n"
        "      schema:\n"
        "        type: object\n"
        "        properties: { ok: { type: boolean } }\n"
        "        required: [ok]\n"
    )
    path = _write(tmp_path, "pipe.yaml", yaml_text)

    from flujo.cli.helpers import validate_pipeline_file

    report = validate_pipeline_file(path)
    assert _count_rule(report, "V-A8") >= 1, "Expected V-A8 before fix"

    from flujo.validation.fixers import apply_fixes_to_file

    applied, backup, metrics = apply_fixes_to_file(path, report, assume_yes=True, rules=["V-A8"])
    assert applied is True
    assert metrics.get("applied", {}).get("V-A8", 0) >= 1

    # Re-validate: V-A8 should be gone
    report2 = validate_pipeline_file(path)
    assert _count_rule(report2, "V-A8") == 0

    # And file contains the field
    text2 = Path(path).read_text(encoding="utf-8")
    assert "structured_output: openai_json" in text2


def test_fix_va8_patch_preview(tmp_path: Path) -> None:
    yaml_text = (
        'version: "0.1"\n'
        "steps:\n"
        "  - kind: step\n"
        "    name: Ex\n"
        "    agent: { id: 'flujo.builtins.stringify' }\n"
        "    processing:\n"
        "      schema:\n"
        "        type: object\n"
        "        properties: { a: { type: string } }\n"
    )
    path = _write(tmp_path, "p.yaml", yaml_text)

    from flujo.cli.helpers import validate_pipeline_file
    from flujo.validation.fixers import build_fix_patch

    report = validate_pipeline_file(path)
    patch, metrics = build_fix_patch(path, report, rules=["V-A8"])
    assert metrics.get("total_applied", 0) >= 1
    assert "structured_output: openai_json" in patch
