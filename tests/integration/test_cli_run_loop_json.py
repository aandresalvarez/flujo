import os
import json
from typer.testing import CliRunner
from flujo.cli.main import app

# Locate the examples directory relative to this file
tests_dir = os.path.dirname(__file__)
EXAMPLES_DIR = os.path.abspath(os.path.join(tests_dir, os.pardir, "examples"))
PIPELINE_FILE = os.path.join(EXAMPLES_DIR, "07_loop_step.py")

runner = CliRunner()


def test_cli_run_loop_step_json_nested_history():
    """
    The CLI `run` command in JSON mode should include nested iteration history
    for a LoopStep in its `step_history` structure.
    """
    result = runner.invoke(
        app,
        ["run", PIPELINE_FILE, "--input", "Write a sentence about Python.", "--json"],
    )
    assert result.exit_code == 0, (
        f"CLI exited non-zero: {result.exit_code}\n{result.stdout}\n{result.stderr}"
    )
    # Parse JSON output
    data = json.loads(result.stdout)
    # Find the loop step by name
    names = [step["name"] for step in data.get("step_history", [])]
    assert "IterativeRefinementLoop" in names, f"Loop step missing, got: {names}"
    idx = names.index("IterativeRefinementLoop")
    loop_entry = data["step_history"][idx]
    # Verify nested iterations exist
    nested = loop_entry.get("step_history")
    assert isinstance(nested, list) and len(nested) >= 2, f"Expected >=2 iterations, got: {nested}"
    # Check names of nested steps
    assert nested[0]["name"] == "EditAndReview"
    assert nested[1]["name"] == "EditAndReview"
