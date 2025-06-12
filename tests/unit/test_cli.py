from pydantic_ai_orchestrator.cli.main import app
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
import pytest
import tempfile
import json

runner = CliRunner()

@pytest.fixture
def mock_orchestrator():
    """Fixture to mock the Orchestrator and its methods."""
    with patch("pydantic_ai_orchestrator.cli.main.Orchestrator") as MockOrchestrator:
        mock_instance = MockOrchestrator.return_value
        class DummyCandidate:
            def model_dump(self):
                return {"solution": "mocked", "score": 1.0}
        mock_instance.run.return_value = DummyCandidate()
        yield mock_instance

def test_cli_solve_happy_path(mock_orchestrator):
    result = runner.invoke(app, ["solve", "write a poem"])
    assert result.exit_code == 0
    assert '"solution": "mocked"' in result.stdout
    mock_orchestrator.run_sync.assert_called_once()

def test_cli_bench_command(mock_orchestrator):
    result = runner.invoke(app, ["bench", "test prompt", "--rounds", "2"])
    assert result.exit_code == 0
    assert "Benchmark Complete (2 rounds)" in result.stdout
    assert "Avg latency" in result.stdout
    assert "Avg score" in result.stdout
    assert mock_orchestrator.run.call_count == 2

def test_cli_show_config_masks_secrets(monkeypatch):
    monkeypatch.setenv("ORCH_OPENAI_API_KEY", "sk-secret")
    # This requires re-importing settings or running CLI in a subprocess
    # For simplicity, we'll just check the output format.
    result = runner.invoke(app, ["show-config"])
    assert result.exit_code == 0
    assert "openai_api_key" not in result.stdout
    assert "logfire_api_key" not in result.stdout

def test_cli_version_command():
    result = runner.invoke(app, ["version-cmd"])
    assert result.exit_code == 0
    assert "pydantic-ai-orchestrator version" in result.stdout

def test_cli_solve_with_weights(mock_orchestrator):
    weights = [
        {"item": "Has a docstring", "weight": 0.7},
        {"item": "Includes type hints", "weight": 0.3},
    ]
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        json.dump(weights, f)
        f.flush()
        result = runner.invoke(app, ["solve", "write a poem", "--weights-path", f.name])
    assert result.exit_code == 0
    mock_orchestrator.run.assert_called_once()
    # Check that weights were passed in metadata
    args, kwargs = mock_orchestrator.run.call_args
    task = args[0]
    assert "weights" in task.metadata
    assert task.metadata["weights"] == weights 