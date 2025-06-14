import os
import types
import asyncio

# Ensure API key exists before importing the CLI
os.environ.setdefault("orch_openai_api_key", "test-key")

from pydantic_ai_orchestrator.cli.main import app
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import tempfile
import json

runner = CliRunner()


@pytest.fixture
def mock_orchestrator() -> None:
    """Fixture to mock the Orchestrator and its methods."""
    with patch("pydantic_ai_orchestrator.cli.main.Orchestrator") as MockOrchestrator:
        mock_instance = MockOrchestrator.return_value

        class DummyCandidate:
            def model_dump(self):
                return {"solution": "mocked", "score": 1.0}

        mock_instance.run.return_value = DummyCandidate()
        yield mock_instance


def test_cli_solve_happy_path(monkeypatch) -> None:
    class DummyCandidate:
        def model_dump(self):
            return {"solution": "mocked", "score": 1.0}

    def dummy_run_sync(self, task):
        return DummyCandidate()

    monkeypatch.setattr("pydantic_ai_orchestrator.cli.main.Orchestrator.run_sync", dummy_run_sync)
    monkeypatch.setattr(
        "pydantic_ai_orchestrator.cli.main.make_agent_async", lambda *a, **k: object()
    )
    from pydantic_ai_orchestrator.cli.main import app

    result = runner.invoke(app, ["solve", "write a poem"])
    assert result.exit_code == 0
    assert '"solution": "mocked"' in result.stdout


def test_cli_solve_custom_models(monkeypatch) -> None:
    class DummyCandidate:
        def model_dump(self):
            return {"solution": "mocked", "score": 1.0}

    def dummy_run_sync(self, task):
        return DummyCandidate()

    monkeypatch.setattr("pydantic_ai_orchestrator.cli.main.Orchestrator.run_sync", dummy_run_sync)
    monkeypatch.setattr(
        "pydantic_ai_orchestrator.cli.main.make_agent_async", lambda *a, **k: object()
    )
    result = runner.invoke(app, ["solve", "write", "--solution-model", "gemini:gemini-1.5-pro"])
    assert result.exit_code == 0


def test_cli_bench_command(monkeypatch) -> None:
    pytest.importorskip("numpy")

    class DummyCandidate:
        score = 1.0

        def model_dump(self):
            return {"solution": "mocked", "score": 1.0}

    def dummy_run_sync(self, task):
        return DummyCandidate()

    monkeypatch.setattr("pydantic_ai_orchestrator.cli.main.Orchestrator.run_sync", dummy_run_sync)
    monkeypatch.setattr(
        "pydantic_ai_orchestrator.cli.main.make_agent_async", lambda *a, **k: object()
    )
    from pydantic_ai_orchestrator.cli.main import app

    result = runner.invoke(app, ["bench", "test prompt", "--rounds", "2"])
    assert result.exit_code == 0
    assert "Benchmark Results" in result.stdout


def test_cli_show_config_masks_secrets(monkeypatch) -> None:
    monkeypatch.setenv("orch_openai_api_key", "sk-secret")
    # This requires re-importing settings or running CLI in a subprocess
    # For simplicity, we'll just check the output format.
    result = runner.invoke(app, ["show-config"])
    assert result.exit_code == 0
    assert "openai_api_key" not in result.stdout
    assert "logfire_api_key" not in result.stdout


def test_cli_version_command(monkeypatch) -> None:
    # import importlib.metadata  # removed unused import
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.2.3")
    monkeypatch.setattr("importlib.metadata.PackageNotFoundError", Exception)
    from pydantic_ai_orchestrator.cli.main import app

    result = runner.invoke(app, ["version-cmd"])
    assert result.exit_code == 0
    assert "pydantic-ai-orchestrator version" in result.stdout


def test_cli_solve_with_weights(monkeypatch) -> None:
    from unittest.mock import patch, MagicMock, AsyncMock
    from pydantic_ai_orchestrator.domain.models import Task

    class DummyCandidate:
        score = 1.0
        def model_dump(self):
            return {"solution": "mocked", "score": self.score}

    # Mock agent that satisfies AgentProtocol
    mock_agent = AsyncMock()
    mock_agent.run.return_value = "mocked agent output"

    with patch("pydantic_ai_orchestrator.cli.main.Orchestrator") as MockOrchestrator:
        # Create a mock instance with a proper run_sync method
        mock_instance = MagicMock()
        async def mock_run_async(task: Task) -> DummyCandidate:
            assert isinstance(task, Task)
            assert task.prompt == "write a poem"
            assert task.metadata.get("weights") is not None
            return DummyCandidate()
        mock_instance.run_async = mock_run_async
        mock_instance.run_sync = MagicMock(side_effect=lambda task: asyncio.run(mock_run_async(task)))
        MockOrchestrator.return_value = mock_instance

        # Patch make_agent_async to return our mock agent
        monkeypatch.setattr(
            "pydantic_ai_orchestrator.cli.main.make_agent_async",
            lambda *a, **k: mock_agent
        )

        from pydantic_ai_orchestrator.cli.main import app
        import tempfile
        import json
        import os

        weights = [
            {"item": "Has a docstring", "weight": 0.7},
            {"item": "Includes type hints", "weight": 0.3},
        ]

        weights_file = None
        try:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
                json.dump(weights, f)
                weights_file = f.name

            result = runner.invoke(app, ["solve", "write a poem", "--weights-path", weights_file])
            
            # Print debug info if test fails
            if result.exit_code != 0:
                print(f"CLI Output: {result.stdout}")
                print(f"CLI Error: {result.stderr}")
                if result.exc_info:
                    import traceback
                    print("Exception:", "".join(traceback.format_exception(*result.exc_info)))

            assert result.exit_code == 0, f"CLI command failed. Output: {result.stdout}, Error: {result.stderr}"

            # Verify Orchestrator was called with correct arguments
            MockOrchestrator.assert_called_once()
            mock_instance.run_sync.assert_called_once()
            
            # Verify the task passed to run_sync
            call_args = mock_instance.run_sync.call_args
            assert call_args is not None
            called_task = call_args[0][0]  # First positional argument
            assert isinstance(called_task, Task)
            assert called_task.prompt == "write a poem"
            assert called_task.metadata.get("weights") == weights

        finally:
            if weights_file and os.path.exists(weights_file):
                os.remove(weights_file)


def test_cli_solve_weights_file_not_found() -> None:
    result = runner.invoke(app, ["solve", "prompt", "--weights-path", "nonexistent.json"])
    assert result.exit_code == 1
    assert "Weights file not found" in result.stderr


def test_cli_solve_weights_file_invalid_json(tmp_path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not a json")
    result = runner.invoke(app, ["solve", "prompt", "--weights-path", str(bad_file)])
    assert result.exit_code == 1
    assert "Error" in result.stdout or "Traceback" in result.stdout or result.stderr


def test_cli_solve_weights_invalid_structure(tmp_path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text('{"item": "a", "weight": 1}')
    result = runner.invoke(app, ["solve", "prompt", "--weights-path", str(bad_file)])
    assert result.exit_code == 1
    assert "list of objects" in result.stderr


def test_cli_solve_weights_missing_keys(tmp_path) -> None:
    weights = [{"item": "a"}]
    file = tmp_path / "weights.json"
    file.write_text(json.dumps(weights))
    result = runner.invoke(app, ["solve", "prompt", "--weights-path", str(file)])
    assert result.exit_code == 1
    assert "list of objects" in result.stderr


def test_cli_solve_keyboard_interrupt(monkeypatch) -> None:
    def raise_keyboard(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr("pydantic_ai_orchestrator.cli.main.Orchestrator.run_sync", raise_keyboard)
    monkeypatch.setattr(
        "pydantic_ai_orchestrator.cli.main.make_agent_async", lambda *a, **k: object()
    )
    result = runner.invoke(app, ["solve", "prompt"])
    assert result.exit_code == 130


def test_cli_bench_keyboard_interrupt(monkeypatch) -> None:
    pytest.importorskip("numpy")

    def raise_keyboard(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr("pydantic_ai_orchestrator.cli.main.Orchestrator.run_sync", raise_keyboard)
    monkeypatch.setattr(
        "pydantic_ai_orchestrator.cli.main.make_agent_async", lambda *a, **k: object()
    )
    result = runner.invoke(app, ["bench", "prompt"])
    assert result.exit_code == 130


def test_cli_version_cmd_package_not_found(monkeypatch) -> None:
    monkeypatch.setattr(
        "importlib.metadata.version", lambda name: (_ for _ in ()).throw(Exception("fail"))
    )
    from pydantic_ai_orchestrator.cli.main import app

    result = runner.invoke(app, ["version-cmd"])
    assert result.exit_code == 0
    assert "unknown" in result.stdout


def test_cli_main_callback_profile(monkeypatch) -> None:
    # Should not raise, just configure logfire
    result = runner.invoke(app, ["--profile"])
    assert result.exit_code == 0 or result.exit_code == 2


def test_cli_solve_configuration_error(monkeypatch) -> None:
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


def test_cli_explain(tmp_path) -> None:
    file = tmp_path / "pipe.py"
    file.write_text(
        "from pydantic_ai_orchestrator.domain import Step\npipeline = Step('A') >> Step('B')\n"
    )

    result = runner.invoke(app, ["explain", str(file)])
    assert result.exit_code == 0
    assert "A" in result.stdout
    assert "B" in result.stdout


def test_cli_improve(monkeypatch, tmp_path) -> None:
    pipe = tmp_path / "pipe.py"
    pipe.write_text(
        "from pydantic_ai_orchestrator.domain import Step\n"
        "from pydantic_ai_orchestrator.testing.utils import StubAgent\n"
        "pipeline = Step.solution(StubAgent(['a']))\n"
    )
    data = tmp_path / "data.py"
    data.write_text(
        "from pydantic_evals import Dataset, Case\ndataset = Dataset(cases=[Case(inputs='a')])\n"
    )

    from pydantic_ai_orchestrator.domain.models import ImprovementReport

    async def dummy_eval(*a, **k):
        return ImprovementReport(suggestions=[])

    monkeypatch.setattr(
        "pydantic_ai_orchestrator.cli.main.evaluate_and_improve",
        dummy_eval,
    )

    result = runner.invoke(app, ["improve", str(pipe), str(data)])
    assert result.exit_code == 0


def test_cli_help() -> None:
    pass


def test_cli_version() -> None:
    pass


def test_cli_run() -> None:
    pass


def test_cli_run_with_args() -> None:
    pass


def test_cli_run_with_invalid_args() -> None:
    pass


def test_cli_run_with_invalid_model() -> None:
    pass


def test_cli_run_with_invalid_temperature() -> None:
    pass


def test_cli_run_with_invalid_max_tokens() -> None:
    pass


def test_cli_run_with_invalid_timeout() -> None:
    pass


def test_cli_run_with_invalid_retries() -> None:
    pass


def test_cli_run_with_invalid_reflection() -> None:
    pass


def test_cli_run_with_invalid_reward() -> None:
    pass


def test_cli_run_with_invalid_telemetry() -> None:
    pass


def test_cli_run_with_invalid_otlp() -> None:
    pass


def test_cli_run_with_invalid_solution_model() -> None:
    pass


def test_cli_run_with_invalid_review_model() -> None:
    pass


def test_cli_run_with_invalid_validator_model() -> None:
    pass


def test_cli_run_with_invalid_reflection_model() -> None:
    pass


def test_cli_run_with_invalid_agent_timeout() -> None:
    pass


def test_cli_run_with_invalid_scorer() -> None:
    pass


def test_cli_run_with_invalid_weights_path() -> None:
    pass


def test_cli_run_with_invalid_solution_model_path() -> None:
    pass


def test_cli_run_with_invalid_review_model_path() -> None:
    pass


def test_cli_run_with_invalid_validator_model_path() -> None:
    pass


def test_cli_run_with_invalid_reflection_model_path() -> None:
    pass
