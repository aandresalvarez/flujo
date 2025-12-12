from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import json


def test_json_includes_fixes_metrics_and_dry_run(tmp_path: Path) -> None:
    yml = (
        'version: "0.1"\n'
        "agents:\n"
        '  typed: { id: "tests.unit.test_error_messages.need_str", model: "local:mock", system_prompt: "typed", output_schema: { type: string } }\n'
        "steps:\n"
        '  - name: A\n    uses: agents.typed\n    input: "hello"\n'
        '  - name: B\n    uses: agents.typed\n    input_schema: { type: string }\n    input: "{{ previous_step.output }} {{ previous_step | to_json }}"\n'
    )
    f = tmp_path / "p.yaml"
    f.write_text(yml)
    res = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "validate",
            str(f),
            "--fix",
            "--fix-rules",
            "V-T1,V-T3",
            "--format=json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(res.stdout or "{}")
    assert "fixes" in data and isinstance(data["fixes"], dict)

    res2 = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "flujo.cli.main",
            "validate",
            str(f),
            "--fix",
            "--fix-dry-run",
            "--fix-rules",
            "V-T1,V-T3",
            "--format=json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    d2 = json.loads(res2.stdout or "{}")
    assert d2.get("fixes_dry_run") is True


def test_json_counts_with_telemetry(tmp_path: Path) -> None:
    yml = (
        'version: "0.1"\n'
        "agents:\n"
        '  typed: { id: "tests.unit.test_error_messages.need_str", model: "local:mock", system_prompt: "typed", output_schema: { type: string } }\n'
        "steps:\n"
        '  - name: A\n    uses: agents.typed\n    input: "hello"\n'
    )
    f = tmp_path / "p.yaml"
    f.write_text(yml)
    env = os_environ_copy()
    env["FLUJO_CLI_TELEMETRY"] = "1"
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "flujo.cli.main", "validate", str(f), "--format=json"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    data = json.loads(res.stdout or "{}")
    counts = data.get("counts") or {}
    assert isinstance(counts, dict)
    assert "warning" in counts and isinstance(counts["warning"], dict)


def test_telemetry_counts_exact_v_t1(tmp_path: Path) -> None:
    yml = (
        'version: "0.1"\n'
        "agents:\n"
        '  typed: { id: "tests.unit.test_error_messages.need_str", model: "local:mock", system_prompt: "typed", output_schema: { type: string } }\n'
        "steps:\n"
        '  - name: A\n    uses: agents.typed\n    input: "hello"\n'
        '  - name: B\n    uses: agents.typed\n    input_schema: { type: string }\n    input: "{{ previous_step.output }}"\n'
    )
    f = tmp_path / "a.yaml"
    f.write_text(yml)
    env = os_environ_copy()
    env["FLUJO_CLI_TELEMETRY"] = "1"
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "flujo.cli.main", "validate", str(f), "--format=json"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    data = json.loads(res.stdout or "{}")
    counts = data.get("counts") or {}
    warnings = counts.get("warning") or {}
    assert warnings.get("V-T1") == 1


def os_environ_copy() -> dict:
    import os

    return dict(os.environ)
