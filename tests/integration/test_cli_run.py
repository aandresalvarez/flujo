import os
from typer.testing import CliRunner
from flujo.cli.main import app

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "../../examples")
PIPELINE_FILE = os.path.join(EXAMPLES_DIR, "test_pipeline.py")
CONTEXT_JSON = os.path.join(EXAMPLES_DIR, "test_context.json")
CONTEXT_YAML = os.path.join(EXAMPLES_DIR, "test_context.yaml")

runner = CliRunner()


def test_run_basic():
    result = runner.invoke(
        app,
        [
            "run",
            PIPELINE_FILE,
            "--input",
            "Test input",
            "--context-model",
            "TestContext",
        ],
    )
    assert result.exit_code == 0
    assert "Pipeline execution completed successfully!" in result.output
    assert "Final output:" in result.output
    assert "Test input" in result.output
    assert 'counter": 3' in result.output  # context counter incremented


def test_run_with_context_json():
    result = runner.invoke(
        app,
        [
            "run",
            PIPELINE_FILE,
            "--input",
            "From JSON",
            "--context-model",
            "TestContext",
            "--context-file",
            CONTEXT_JSON,
        ],
    )
    assert result.exit_code == 0
    assert "Final output:" in result.output
    assert "From JSON" in result.output
    assert 'counter": 13' in result.output  # 10 + 3 steps
    assert "Initial message" in result.output


def test_run_with_context_yaml():
    result = runner.invoke(
        app,
        [
            "run",
            PIPELINE_FILE,
            "--input",
            "From YAML",
            "--context-model",
            "TestContext",
            "--context-file",
            CONTEXT_YAML,
        ],
    )
    assert result.exit_code == 0
    assert "Final output:" in result.output
    assert "From YAML" in result.output
    assert 'counter": 8' in result.output  # 5 + 3 steps
    assert "YAML initial message" in result.output


def test_run_missing_pipeline():
    result = runner.invoke(
        app,
        ["run", "nonexistent.py", "--input", "fail", "--context-model", "TestContext"],
    )
    assert result.exit_code != 0
    assert "Failed to load pipeline file" in result.output or "No" in result.output


def test_run_missing_context_model():
    result = runner.invoke(
        app,
        ["run", PIPELINE_FILE, "--input", "fail", "--context-model", "DoesNotExist"],
    )
    assert result.exit_code != 0
    assert "Context model" in result.output
