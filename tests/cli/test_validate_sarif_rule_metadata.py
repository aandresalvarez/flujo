from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import json


def test_sarif_includes_rule_metadata(tmp_path: Path) -> None:
    yml = (
        'version: "0.1"\n'
        "steps:\n"
        '  - name: A\n    agent: { id: "flujo.builtins.stringify" }\n    input: "{{ previous_step.output }}"\n'
    )
    f = tmp_path / "p.yaml"
    f.write_text(yml)
    res = subprocess.run(
        [sys.executable, "-m", "flujo.cli.main", "validate", str(f), "--format=sarif"],
        capture_output=True,
        text=True,
    )
    sarif = json.loads(res.stdout or "{}")
    runs = sarif.get("runs") or []
    assert runs and isinstance(runs[0], dict)
    drv = runs[0].get("tool", {}).get("driver", {})
    rules = drv.get("rules") or []
    assert any(r.get("id") == "V-T1" for r in rules)
    # All rules should include helpUri and a name; shortDescription.text recommended; fullDescription.text preferred
    assert all("helpUri" in r for r in rules)
    assert all("name" in r for r in rules)
    # At least one rule should include a shortDescription and fullDescription
    assert any(isinstance(r.get("shortDescription", {}).get("text"), str) for r in rules)
    assert any(isinstance(r.get("fullDescription", {}).get("text"), str) for r in rules)
