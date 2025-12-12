from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write(tmp: Path, name: str, text: str) -> Path:
    p = tmp / name
    p.write_text(text)
    return p


def test_validate_sarif_contains_rules_and_results(tmp_path: Path) -> None:
    yml = (
        'version: "0.1"\n'
        "agents:\n"
        '  typed: { id: "tests.unit.test_error_messages.need_str", model: "local:mock", system_prompt: "typed", output_schema: { type: string } }\n'
        "steps:\n"
        '  - name: A\n    uses: agents.typed\n    input: "hello"\n'
        '  - name: B\n    uses: agents.typed\n    input_schema: { type: string }\n    input: "{{ previous_step.output }}"\n'
    )
    f = _write(tmp_path, "p.yaml", yml)
    res = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "validate", str(f), "--format=sarif"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    payload = json.loads(res.stdout or "{}")
    assert payload.get("version") == "2.1.0"
    runs = payload.get("runs") or []
    assert runs and runs[0].get("tool", {}).get("driver", {}).get("name") == "flujo-validate"
    results = runs[0].get("results") or []
    assert any(r.get("ruleId") in {"V-T1", "V-S1", "V-A1", "V-A5"} for r in results)


def test_validate_rules_profile_strict_makes_vt1_error(tmp_path: Path) -> None:
    # flujo.toml with a strict profile that escalates template rules to error
    flujo_toml = '[validation.profiles.strict]\n"V-T*" = "error"\n'
    _write(tmp_path, "flujo.toml", flujo_toml)
    yml = (
        'version: "0.1"\n'
        "agents:\n"
        '  typed: { id: "tests.unit.test_error_messages.need_str", model: "local:mock", system_prompt: "typed", output_schema: { type: string } }\n'
        "steps:\n"
        '  - name: A\n    uses: agents.typed\n    input: "hello"\n'
        '  - name: B\n    uses: agents.typed\n    input: "{{ previous_step.missing_field }}"\n'
    )
    f = _write(tmp_path, "p.yaml", yml)
    import os

    env = os.environ.copy()
    env["FLUJO_CONFIG_PATH"] = str(tmp_path / "flujo.toml")
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "validate",
            str(f),
            "--format=json",
            "--rules=strict",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    # With V-T* = error, strict default should fail
    assert res.returncode == 4
    payload = json.loads(res.stdout or "{}")
    assert any(e.get("rule_id") in {"V-T1", "V-T5"} for e in payload.get("errors") or [])


def test_validate_rules_file_off_suppresses_findings(tmp_path: Path) -> None:
    rules = {"V-T*": "off"}
    _write(tmp_path, "rules.json", json.dumps(rules))
    yml = (
        'version: "0.1"\n'
        "steps:\n"
        '  - name: A\n    agent: { id: "flujo.builtins.echo" }\n    meta: { is_adapter: true, adapter_id: generic-adapter, adapter_allow: generic }\n    input: "hello"\n'
        '  - name: B\n    agent: { id: "flujo.builtins.stringify" }\n    meta: { is_adapter: true, adapter_id: generic-adapter, adapter_allow: generic }\n    input: "{{ previous_step.output }}"\n'
    )
    f = _write(tmp_path, "p.yaml", yml)
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "validate",
            str(f),
            "--format=json",
            "--rules",
            str(tmp_path / "rules.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    payload = json.loads(res.stdout or "{}")
    assert all(w.get("rule_id") != "V-T1" for w in payload.get("warnings") or [])


def test_validate_baseline_deltas_and_update(tmp_path: Path) -> None:
    # Start with a pipeline that has a single V-T1
    yml1 = (
        'version: "0.1"\n'
        "agents:\n"
        '  typed: { id: "tests.unit.test_error_messages.need_str", model: "local:mock", system_prompt: "typed", output_schema: { type: string } }\n'
        "steps:\n"
        '  - name: A\n    uses: agents.typed\n    input: "hello"\n'
        '  - name: B\n    uses: agents.typed\n    input: "{{ previous_step.output }}"\n'
    )
    f = _write(tmp_path, "p.yaml", yml1)
    # First run, capture JSON and write baseline
    res1 = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "validate", str(f), "--format=json"],
        capture_output=True,
        text=True,
    )
    assert res1.returncode in (0, 4)
    payload1 = json.loads(res1.stdout or "{}")
    baseline = tmp_path / "baseline.json"
    _write(
        tmp_path,
        "baseline.json",
        json.dumps(
            {
                "errors": payload1.get("errors") or [],
                "warnings": payload1.get("warnings") or [],
            }
        ),
    )

    # Modify pipeline to add an additional finding (another V-T1)
    yml2 = (
        'version: "0.1"\n'
        "agents:\n"
        '  typed: { id: "tests.unit.test_error_messages.need_str", model: "local:mock", system_prompt: "typed", output_schema: { type: string } }\n'
        "steps:\n"
        '  - name: A\n    uses: agents.typed\n    input: "hello"\n'
        '  - name: B\n    uses: agents.typed\n    input_schema: { type: string }\n    input: "{{ previous_step.output }}"\n'
        '  - name: C\n    uses: agents.typed\n    input_schema: { type: string }\n    input: "{{ previous_step.output }}"\n'
    )
    _write(tmp_path, "p.yaml", yml2)
    res2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "validate",
            str(tmp_path / "p.yaml"),
            "--format=json",
            "--baseline",
            str(baseline),
        ],
        capture_output=True,
        text=True,
    )
    assert res2.returncode == 0  # warnings only should not fail strict by default
    payload2 = json.loads(res2.stdout or "{}")
    # Post-baseline view should contain only new findings (at least the added V-T1)
    warns2 = payload2.get("warnings") or []
    assert any(w.get("rule_id") == "V-T1" for w in warns2)
    # Update baseline
    res3 = subprocess.run(
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "validate",
            str(tmp_path / "p.yaml"),
            "--format=json",
            "--baseline",
            str(baseline),
            "--update-baseline",
        ],
        capture_output=True,
        text=True,
    )
    assert res3.returncode == 0
    # Baseline file should now contain only the previously added (new) warnings (step C)
    bdata = json.loads((tmp_path / "baseline.json").read_text())
    assert any(w.get("step_name") == "C" for w in bdata.get("warnings") or [])
    assert all(w.get("step_name") != "A" for w in bdata.get("warnings") or [])
    # Run again; with updated baseline, the other warnings (A/B) are now considered new
    res4 = subprocess.run(
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "validate",
            str(tmp_path / "p.yaml"),
            "--format=json",
            "--baseline",
            str(baseline),
        ],
        capture_output=True,
        text=True,
    )
    payload4 = json.loads(res4.stdout or "{}")
    w4 = payload4.get("warnings") or []
    # With improved V-A5 logic, A's output is considered consumed via templating,
    # so only B's template warning remains new relative to the trimmed baseline.
    assert any(w.get("step_name") == "B" and w.get("rule_id") == "V-T1" for w in w4)


def test_validate_baseline_fail_on_warn_nonzero_for_new(tmp_path: Path) -> None:
    # Baseline with no findings
    (tmp_path / "baseline.json").write_text(json.dumps({"errors": [], "warnings": []}))
    yml = (
        'version: "0.1"\n'
        "agents:\n"
        '  typed: { model: "local:mock", system_prompt: "typed", output_schema: { type: string } }\n'
        "steps:\n"
        '  - name: A\n    uses: agents.typed\n    input: "hello"\n'
        '  - name: B\n    uses: agents.typed\n    input: "{{ previous_step.output }}"\n'
    )
    f = _write(tmp_path, "p.yaml", yml)
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "validate",
            str(f),
            "--format=json",
            "--baseline",
            str(tmp_path / "baseline.json"),
            "--fail-on-warn",
        ],
        capture_output=True,
        text=True,
    )
    # New warnings after baseline -> fail-on-warn should yield EX_VALIDATION_FAILED (4)
    assert res.returncode == 4, res.stdout + res.stderr


def test_validate_baseline_fail_on_warn_zero_when_no_new(tmp_path: Path) -> None:
    # Create pipeline and baseline with its findings
    yml = (
        'version: "0.1"\n'
        "steps:\n"
        '  - name: A\n    agent: { id: "flujo.builtins.stringify" }\n    input: "hello"\n'
        '  - name: B\n    agent: { id: "flujo.builtins.stringify" }\n    input: "{{ previous_step.output }}"\n'
    )
    f = _write(tmp_path, "p.yaml", yml)
    res1 = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "validate", str(f), "--format=json"],
        capture_output=True,
        text=True,
    )
    payload1 = json.loads(res1.stdout or "{}")
    _write(
        tmp_path,
        "baseline.json",
        json.dumps(
            {"errors": payload1.get("errors") or [], "warnings": payload1.get("warnings") or []}
        ),
    )
    # Same pipeline with baseline and --fail-on-warn should exit 0 (no new warnings)
    res2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "validate",
            str(f),
            "--format=json",
            "--baseline",
            str(tmp_path / "baseline.json"),
            "--fail-on-warn",
        ],
        capture_output=True,
        text=True,
    )
    assert res2.returncode == 0, res2.stdout + res2.stderr
