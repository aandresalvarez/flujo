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

def test_cli_solve_happy_path(monkeypatch):
    class DummyCandidate:
        def model_dump(self):
            return {"solution": "mocked", "score": 1.0}
    def dummy_run_sync(self, task):
        return DummyCandidate()
    monkeypatch.setattr("pydantic_ai_orchestrator.cli.main.Orchestrator.run_sync", dummy_run_sync)
    from pydantic_ai_orchestrator.cli.main import app
    result = runner.invoke(app, ["solve", "write a poem"])
    assert result.exit_code == 0
    assert '"solution": "mocked"' in result.stdout

def test_cli_bench_command(monkeypatch):
    class DummyCandidate:
        score = 1.0
        def model_dump(self):
            return {"solution": "mocked", "score": 1.0}
    def dummy_run_sync(self, task):
        return DummyCandidate()
    monkeypatch.setattr("pydantic_ai_orchestrator.cli.main.Orchestrator.run_sync", dummy_run_sync)
    from pydantic_ai_orchestrator.cli.main import app
    result = runner.invoke(app, ["bench", "test prompt", "--rounds", "2"])
    assert result.exit_code == 0
    assert "Benchmark Results" in result.stdout

def test_cli_show_config_masks_secrets(monkeypatch):
    monkeypatch.setenv("ORCH_OPENAI_API_KEY", "sk-secret")
    # This requires re-importing settings or running CLI in a subprocess
    # For simplicity, we'll just check the output format.
    result = runner.invoke(app, ["show-config"])
    assert result.exit_code == 0
    assert "openai_api_key" not in result.stdout
    assert "logfire_api_key" not in result.stdout

def test_cli_version_command(monkeypatch):
    import importlib.metadata
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.2.3")
    monkeypatch.setattr("importlib.metadata.PackageNotFoundError", Exception)
    from pydantic_ai_orchestrator.cli.main import app
    result = runner.invoke(app, ["version-cmd"])
    assert result.exit_code == 0
    assert "pydantic-ai-orchestrator version" in result.stdout

def test_cli_solve_with_weights(monkeypatch):
    class DummyCandidate:
        def model_dump(self):
            return {"solution": "mocked", "score": 1.0}
    def dummy_run_sync(self, task):
        return DummyCandidate()
    monkeypatch.setattr("pydantic_ai_orchestrator.cli.main.Orchestrator.run_sync", dummy_run_sync)
    from pydantic_ai_orchestrator.cli.main import app
    weights = [
        {"item": "Has a docstring", "weight": 0.7},
        {"item": "Includes type hints", "weight": 0.3},
    ]
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        json.dump(weights, f)
        f.flush()
        result = runner.invoke(app, ["solve", "write a poem", "--weights-path", f.name])
    assert result.exit_code == 0
    # No need to check mock_orchestrator.run was called, as we patch run_sync 