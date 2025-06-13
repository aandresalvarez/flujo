import os

# Ensure API key exists before importing the CLI
os.environ.setdefault("orch_openai_api_key", "test-key")

from pydantic_ai_orchestrator.cli.main import app
from typer.testing import CliRunner
from unittest.mock import patch
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
    monkeypatch.setattr("pydantic_ai_orchestrator.cli.main.make_agent_async", lambda *a, **k: object())
    from pydantic_ai_orchestrator.cli.main import app
    result = runner.invoke(app, ["solve", "write a poem"])
    assert result.exit_code == 0
    assert '"solution": "mocked"' in result.stdout

def test_cli_solve_custom_models(monkeypatch):
    class DummyCandidate:
        def model_dump(self):
            return {"solution": "mocked", "score": 1.0}

    def dummy_run_sync(self, task):
        return DummyCandidate()

    monkeypatch.setattr("pydantic_ai_orchestrator.cli.main.Orchestrator.run_sync", dummy_run_sync)
    monkeypatch.setattr("pydantic_ai_orchestrator.cli.main.make_agent_async", lambda *a, **k: object())
    result = runner.invoke(app, ["solve", "write", "--solution-model", "gemini:gemini-1.5-pro"])
    assert result.exit_code == 0

def test_cli_bench_command(monkeypatch):
    pytest.importorskip("numpy")
    class DummyCandidate:
        score = 1.0
        def model_dump(self):
            return {"solution": "mocked", "score": 1.0}
    def dummy_run_sync(self, task):
        return DummyCandidate()
    monkeypatch.setattr("pydantic_ai_orchestrator.cli.main.Orchestrator.run_sync", dummy_run_sync)
    monkeypatch.setattr("pydantic_ai_orchestrator.cli.main.make_agent_async", lambda *a, **k: object())
    from pydantic_ai_orchestrator.cli.main import app
    result = runner.invoke(app, ["bench", "test prompt", "--rounds", "2"])
    assert result.exit_code == 0
    assert "Benchmark Results" in result.stdout

def test_cli_show_config_masks_secrets(monkeypatch):
    monkeypatch.setenv("orch_openai_api_key", "sk-secret")
    # This requires re-importing settings or running CLI in a subprocess
    # For simplicity, we'll just check the output format.
    result = runner.invoke(app, ["show-config"])
    assert result.exit_code == 0
    assert "openai_api_key" not in result.stdout
    assert "logfire_api_key" not in result.stdout

def test_cli_version_command(monkeypatch):
    # import importlib.metadata  # removed unused import
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
    monkeypatch.setattr("pydantic_ai_orchestrator.cli.main.make_agent_async", lambda *a, **k: object())
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

def test_cli_solve_weights_file_not_found():
    result = runner.invoke(app, ["solve", "prompt", "--weights-path", "nonexistent.json"])
    assert result.exit_code == 1
    assert "Weights file not found" in result.stderr

def test_cli_solve_weights_file_invalid_json(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not a json")
    result = runner.invoke(app, ["solve", "prompt", "--weights-path", str(bad_file)])
    assert result.exit_code == 1
    assert "Error" in result.stdout or "Traceback" in result.stdout or result.stderr

def test_cli_solve_weights_invalid_structure(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{\"item\": \"a\", \"weight\": 1}")
    result = runner.invoke(app, ["solve", "prompt", "--weights-path", str(bad_file)])
    assert result.exit_code == 1
    assert "list of objects" in result.stderr

def test_cli_solve_weights_missing_keys(tmp_path):
    weights = [{"item": "a"}]
    file = tmp_path / "weights.json"
    file.write_text(json.dumps(weights))
    result = runner.invoke(app, ["solve", "prompt", "--weights-path", str(file)])
    assert result.exit_code == 1
    assert "list of objects" in result.stderr

def test_cli_solve_keyboard_interrupt(monkeypatch):
    def raise_keyboard(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr("pydantic_ai_orchestrator.cli.main.Orchestrator.run_sync", raise_keyboard)
    monkeypatch.setattr("pydantic_ai_orchestrator.cli.main.make_agent_async", lambda *a, **k: object())
    result = runner.invoke(app, ["solve", "prompt"])
    assert result.exit_code == 130

def test_cli_bench_keyboard_interrupt(monkeypatch):
    pytest.importorskip("numpy")
    def raise_keyboard(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr("pydantic_ai_orchestrator.cli.main.Orchestrator.run_sync", raise_keyboard)
    monkeypatch.setattr("pydantic_ai_orchestrator.cli.main.make_agent_async", lambda *a, **k: object())
    result = runner.invoke(app, ["bench", "prompt"])
    assert result.exit_code == 130

def test_cli_version_cmd_package_not_found(monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda name: (_ for _ in ()).throw(Exception("fail")))
    from pydantic_ai_orchestrator.cli.main import app
    result = runner.invoke(app, ["version-cmd"])
    assert result.exit_code == 0
    assert "unknown" in result.stdout

def test_cli_main_callback_profile(monkeypatch):
    # Should not raise, just configure logfire
    result = runner.invoke(app, ["--profile"])
    assert result.exit_code == 0 or result.exit_code == 2


def test_cli_solve_configuration_error(monkeypatch):
    """Test that configuration errors surface with exit code 2."""

    def raise_config_error(*args, **kwargs):
        from pydantic_ai_orchestrator.exceptions import ConfigurationError
        raise ConfigurationError("Missing API key!")

    monkeypatch.setattr(
        "pydantic_ai_orchestrator.cli.main.make_agent_async",
        raise_config_error,
    )

    result = runner.invoke(app, ["solve", "prompt"])
    assert result.exit_code == 2
    assert "Configuration Error: Missing API key!" in result.stderr

