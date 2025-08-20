from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


def test_externalized_prompt_cli_compiles_from_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Prepare project structure inside tmp_path
    base_dir = tmp_path
    prompts_dir = base_dir / "shared_prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    # Write prompt file explicitly with UTF-8 encoding
    (prompts_dir / "agent_prompt.md").write_text("Hello from prompt file.\n", encoding="utf-8")

    # Minimal YAML that references the externalized prompt within base_dir
    yaml_text = (
        'version: "0.1"\n'
        "agents:\n"
        "  a1:\n"
        '    model: "openai:gpt-4o"\n'
        "    system_prompt:\n"
        '      from_file: "./shared_prompts/agent_prompt.md"\n'
        '    output_schema: { type: "object" }\n'
        "steps:\n"
        "  - kind: step\n"
        "    name: S1\n"
        "    uses: agents.a1\n"
    )

    yaml_file = base_dir / "pipeline.yaml"
    yaml_file.write_text(yaml_text, encoding="utf-8")

    # Ensure API key present to allow agent construction during compile
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Load via Python API and assert structure
    from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml
    from flujo.domain.dsl import Pipeline

    compiled = load_pipeline_blueprint_from_yaml(
        yaml_file.read_text(encoding="utf-8"), base_dir=str(base_dir)
    )
    assert isinstance(compiled, Pipeline)
    assert compiled.steps and len(compiled.steps) >= 1

    # Also verify a subprocess can import and compile the pipeline from base_dir
    code = (
        "from pathlib import Path\n"
        "from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml\n"
        f"p = Path(r'{str(yaml_file)}')\n"
        "text = p.read_text(encoding='utf-8')\n"
        f"_ = load_pipeline_blueprint_from_yaml(text, base_dir=r'{str(base_dir)}')\n"
        "print('OK')\n"
    )
    try:
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"Subprocess timed out: {e}")

    if res.returncode != 0:
        pytest.fail(f"Subprocess failed: {res.stderr}")
    assert "OK" in res.stdout
