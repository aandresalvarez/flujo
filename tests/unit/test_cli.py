import asyncio
from typing import Any

# Tests require an API key; a fixture sets it for each test

from flujo.cli.main import app
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import json

runner = CliRunner()


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch) -> None:
    """Ensure OPENAI_API_KEY is present and refresh settings for each test."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import flujo.infra.settings as settings_mod

    new_settings = settings_mod.Settings()
    monkeypatch.setattr(settings_mod, "settings", new_settings, raising=False)
    monkeypatch.setattr("flujo.infra.agents.settings", new_settings, raising=False)
    monkeypatch.setattr("flujo.application.temperature.settings", new_settings, raising=False)
    monkeypatch.setattr("flujo.cli.main.settings", new_settings, raising=False)


@pytest.fixture
def mock_orchestrator() -> None:
    """Fixture to mock the Default and its methods."""
    with patch("flujo.cli.main.Default") as MockDefault:
        mock_instance = MockDefault.return_value

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

    monkeypatch.setattr("flujo.cli.main.Default.run_sync", dummy_run_sync)
    monkeypatch.setattr("flujo.cli.main.make_agent_async", lambda *a, **k: object())
    from flujo.cli.main import app

    result = runner.invoke(app, ["solve", "write a poem"])
    assert result.exit_code == 0
    assert '"solution": "mocked"' in result.stdout


def test_cli_solve_custom_models(monkeypatch) -> None:
    class DummyCandidate:
        def model_dump(self):
            return {"solution": "mocked", "score": 1.0}

    def dummy_run_sync(self, task):
        return DummyCandidate()

    monkeypatch.setattr("flujo.cli.main.Default.run_sync", dummy_run_sync)
    monkeypatch.setattr("flujo.cli.main.make_agent_async", lambda *a, **k: object())
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

    monkeypatch.setattr("flujo.cli.main.Default.run_sync", dummy_run_sync)
    monkeypatch.setattr("flujo.cli.main.make_agent_async", lambda *a, **k: object())
    from flujo.cli.main import app

    result = runner.invoke(app, ["bench", "test prompt", "--rounds", "2"])
    assert result.exit_code == 0
    assert "Benchmark Results" in result.stdout


def test_cli_show_config_masks_secrets(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
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
    from flujo.cli.main import app

    result = runner.invoke(app, ["version-cmd"])
    assert result.exit_code == 0
    assert "flujo version" in result.stdout


def test_cli_solve_with_weights(monkeypatch) -> None:
    from unittest.mock import patch
    from flujo.domain.models import Task

    class DummyCandidate:
        score = 1.0

        def model_dump(self):
            return {"solution": "mocked", "score": self.score}

    # Mock agent that satisfies AgentProtocol
    mock_agent = AsyncMock()
    mock_agent.run.return_value = "mocked agent output"

    with patch("flujo.cli.main.Default") as MockDefault:
        # Create a mock instance with a proper run_sync method
        mock_instance = MagicMock()

        async def mock_run_async(task: Task) -> DummyCandidate:
            assert isinstance(task, Task)
            assert task.prompt == "write a poem"
            assert task.metadata.get("weights") is not None
            return DummyCandidate()

        mock_instance.run_async = mock_run_async
        mock_instance.run_sync = MagicMock(
            side_effect=lambda task: asyncio.run(mock_run_async(task))
        )
        MockDefault.return_value = mock_instance

        # Patch make_agent_async to return our mock agent
        monkeypatch.setattr("flujo.cli.main.make_agent_async", lambda *a, **k: mock_agent)

        from flujo.cli.main import app
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

            assert result.exit_code == 0, (
                f"CLI command failed. Output: {result.stdout}, Error: {result.stderr}"
            )

            # Verify Default was called with correct arguments
            MockDefault.assert_called_once()
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

    monkeypatch.setattr("flujo.cli.main.Default.run_sync", raise_keyboard)
    monkeypatch.setattr("flujo.cli.main.make_agent_async", lambda *a, **k: object())
    result = runner.invoke(app, ["solve", "prompt"])
    assert result.exit_code == 130


def test_cli_bench_keyboard_interrupt(monkeypatch) -> None:
    pytest.importorskip("numpy")

    def raise_keyboard(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr("flujo.cli.main.Default.run_sync", raise_keyboard)
    monkeypatch.setattr("flujo.cli.main.make_agent_async", lambda *a, **k: object())
    result = runner.invoke(app, ["bench", "prompt"])
    assert result.exit_code == 130


def test_cli_version_cmd_package_not_found(monkeypatch) -> None:
    monkeypatch.setattr(
        "importlib.metadata.version", lambda name: (_ for _ in ()).throw(Exception("fail"))
    )
    from flujo.cli.main import app

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
        from flujo.exceptions import ConfigurationError

        raise ConfigurationError("Missing API key!")

    monkeypatch.setattr(
        "flujo.cli.main.make_agent_async",
        raise_config_error,
    )

    result = runner.invoke(app, ["solve", "prompt"])
    assert result.exit_code == 2
    assert "Configuration Error: Missing API key!" in result.stderr


def test_cli_explain(tmp_path) -> None:
    file = tmp_path / "pipe.py"
    file.write_text("from flujo.domain import Step\npipeline = Step('A') >> Step('B')\n")

    result = runner.invoke(app, ["explain", str(file)])
    assert result.exit_code == 0
    assert "A" in result.stdout
    assert "B" in result.stdout


def test_cli_improve_output_formatting(monkeypatch, tmp_path) -> None:
    pipe = tmp_path / "pipe.py"
    pipe.write_text(
        "from flujo.domain import Step\n"
        "from flujo.testing.utils import StubAgent\n"
        "pipeline = Step.solution(StubAgent(['a']))\n"
    )
    data = tmp_path / "data.py"
    data.write_text(
        "from pydantic_evals import Dataset, Case\ndataset = Dataset(cases=[Case(inputs='a')])\n"
    )

    from flujo.domain.models import ImprovementReport

    async def dummy_eval(*a, **k):
        return ImprovementReport(suggestions=[])

    monkeypatch.setattr(
        "flujo.cli.main.evaluate_and_improve",
        dummy_eval,
    )

    result = runner.invoke(app, ["improve", str(pipe), str(data)])
    assert result.exit_code == 0
    assert "IMPROVEMENT REPORT" in result.stdout


def test_cli_improve_json_output(monkeypatch, tmp_path) -> None:
    pipe = tmp_path / "pipe.py"
    pipe.write_text(
        "from flujo.domain import Step\n"
        "from flujo.testing.utils import StubAgent\n"
        "pipeline = Step.solution(StubAgent(['a']))\n"
    )
    data = tmp_path / "data.py"
    data.write_text(
        "from pydantic_evals import Dataset, Case\ndataset = Dataset(cases=[Case(inputs='a')])\n"
    )

    from flujo.domain.models import (
        ImprovementReport,
        ImprovementSuggestion,
        SuggestionType,
        PromptModificationDetail,
    )

    async def dummy_eval(*a, **k):
        return ImprovementReport(
            suggestions=[
                ImprovementSuggestion(
                    target_step_name="A",
                    suggestion_type=SuggestionType.PROMPT_MODIFICATION,
                    failure_pattern_summary="f",
                    detailed_explanation="d",
                    prompt_modification_details=PromptModificationDetail(
                        modification_instruction="m"
                    ),
                )
            ]
        )

    monkeypatch.setattr(
        "flujo.cli.main.evaluate_and_improve",
        dummy_eval,
    )

    result = runner.invoke(app, ["improve", str(pipe), str(data), "--json"])
    assert result.exit_code == 0
    assert '"suggestions"' in result.stdout


def test_cli_help() -> None:
    """Test that the help command works and shows all available commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "solve" in result.stdout
    assert "version-cmd" in result.stdout
    assert "show-config" in result.stdout
    assert "bench" in result.stdout
    assert "improve" in result.stdout
    assert "explain" in result.stdout


def test_cli_version(monkeypatch) -> None:
    """Test that the version command works and shows the correct version."""
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.2.0")
    version = importlib.metadata.version("flujo")
    result = runner.invoke(app, ["version-cmd"])
    assert result.exit_code == 0
    assert version in result.stdout


def test_cli_run() -> None:
    """Test basic run functionality with default settings."""
    from unittest.mock import patch

    class DummyCandidate:
        def model_dump(self):
            return {"solution": "test solution", "score": 1.0}

    with patch("flujo.cli.main.Default") as MockDefault:
        mock_instance = MagicMock()
        mock_instance.run_sync.return_value = DummyCandidate()
        MockDefault.return_value = mock_instance

        result = runner.invoke(app, ["solve", "test prompt"])
        assert result.exit_code == 0
        assert "test solution" in result.stdout
        mock_instance.run_sync.assert_called_once()


def test_cli_run_with_args() -> None:
    """Test run with various command line arguments."""
    from unittest.mock import patch

    class DummyCandidate:
        def model_dump(self):
            return {"solution": "test solution", "score": 1.0}

    dummy_settings = MagicMock()
    dummy_settings.default_solution_model = "gpt-4"
    dummy_settings.default_review_model = "gpt-3.5-turbo"
    dummy_settings.default_validator_model = "gpt-3.5-turbo"
    dummy_settings.default_reflection_model = "gpt-4"
    dummy_settings.reflection_enabled = True
    dummy_settings.scorer = "ratio"
    dummy_settings.reflection_limit = 1

    with (
        patch("flujo.cli.main.Default") as MockDefault,
        patch("flujo.cli.main.make_agent_async") as mock_make_agent,
        patch("flujo.cli.main.get_reflection_agent") as mock_get_reflection_agent,
        patch("flujo.cli.main.settings", dummy_settings),
    ):
        mock_instance = MagicMock()
        mock_instance.run_sync.return_value = DummyCandidate()
        MockDefault.return_value = mock_instance
        mock_make_agent.return_value = MagicMock()
        mock_get_reflection_agent.return_value = MagicMock()

        result = runner.invoke(
            app,
            [
                "solve",
                "test prompt",
                "--max-iters",
                "5",
                "--k",
                "3",
                "--reflection",
                "--scorer",
                "ratio",
                "--solution-model",
                "gpt-4",
                "--review-model",
                "gpt-3.5-turbo",
                "--validator-model",
                "gpt-3.5-turbo",
                "--reflection-model",
                "gpt-4",
            ],
        )
        assert result.exit_code == 0
        assert "test solution" in result.stdout
        mock_instance.run_sync.assert_called_once()


def test_cli_run_with_invalid_args() -> None:
    """Test run with invalid command line arguments."""
    # Test with invalid max-iters
    result = runner.invoke(app, ["solve", "test prompt", "--max-iters", "-1"])
    assert result.exit_code != 0
    assert "Error" in result.stderr

    # Test with invalid k
    result = runner.invoke(app, ["solve", "test prompt", "--k", "0"])
    assert result.exit_code != 0
    assert "Error" in result.stderr

    # Test with invalid scorer
    result = runner.invoke(app, ["solve", "test prompt", "--scorer", "invalid"])
    assert result.exit_code != 0
    assert "Error" in result.stderr


def test_cli_run_with_invalid_model() -> None:
    """Test run with invalid model names."""
    from unittest.mock import patch
    from flujo.exceptions import ConfigurationError

    with patch("flujo.cli.main.make_agent_async") as mock_make_agent:
        mock_make_agent.side_effect = ConfigurationError("Invalid model name")
        result = runner.invoke(app, ["solve", "test prompt", "--solution-model", "invalid-model"])
        assert result.exit_code == 2
        assert "Configuration Error" in result.stderr


def test_cli_run_with_invalid_retries() -> None:
    """Test run with invalid retry settings."""
    from unittest.mock import patch
    from flujo.exceptions import ConfigurationError

    with patch("flujo.cli.main.Default") as MockDefault:
        MockDefault.side_effect = ConfigurationError("Invalid retry settings")
        result = runner.invoke(app, ["solve", "test prompt", "--max-iters", "100"])
        assert result.exit_code == 2
        assert "Configuration Error" in result.stderr


def test_cli_run_with_invalid_review_model() -> None:
    """Test run with invalid review model."""
    from unittest.mock import patch
    from flujo.exceptions import ConfigurationError

    with patch("flujo.cli.main.make_agent_async") as mock_make_agent:
        mock_make_agent.side_effect = ConfigurationError("Invalid review model")
        result = runner.invoke(app, ["solve", "test prompt", "--review-model", "invalid-model"])
        assert result.exit_code == 2
        assert "Configuration Error" in result.stderr


def test_cli_run_with_invalid_agent_timeout() -> None:
    """Test run with invalid agent timeout settings."""
    from unittest.mock import patch
    from flujo.exceptions import ConfigurationError

    with patch("flujo.cli.main.Default") as MockDefault:
        MockDefault.side_effect = ConfigurationError("Invalid agent timeout")
        result = runner.invoke(app, ["solve", "test prompt"])
        assert result.exit_code == 2
        assert "Configuration Error" in result.stderr


def test_cli_run_with_invalid_review_model_path() -> None:
    """Test run with invalid review model path."""
    from unittest.mock import patch
    from flujo.exceptions import ConfigurationError

    with patch("flujo.cli.main.make_agent_async") as mock_make_agent:
        mock_make_agent.side_effect = ConfigurationError("Invalid review model path")
        result = runner.invoke(app, ["solve", "test prompt", "--review-model", "/invalid/path"])
        assert result.exit_code == 2
        assert "Configuration Error" in result.stderr


def test_cli_add_eval_case_prints_correct_case_string(tmp_path) -> None:
    file = tmp_path / "data.py"
    file.write_text("dataset = None")
    result = runner.invoke(
        app,
        [
            "add-eval-case",
            "-d",
            str(file),
            "-n",
            "my_new_test",
            "-i",
            "test input",
            "-e",
            "expected output",
            "--metadata",
            '{"tag":"new"}',
        ],
    )
    assert result.exit_code == 0
    assert 'Case(name="my_new_test"' in result.stdout


def test_cli_add_eval_case_handles_missing_dataset_file_gracefully(tmp_path) -> None:
    missing = tmp_path / "missing.py"
    result = runner.invoke(
        app,
        ["add-eval-case", "-d", str(missing), "-n", "a", "-i", "b", "--expected", ""],
    )
    assert result.exit_code == 1
    assert "Dataset file not found" in result.stdout


def test_cli_add_eval_case_invalid_metadata_json(tmp_path) -> None:
    file = tmp_path / "data.py"
    file.write_text("dataset = None")
    result = runner.invoke(
        app,
        [
            "add-eval-case",
            "-d",
            str(file),
            "-n",
            "case",
            "-i",
            "x",
            "--metadata",
            "{not json}",
            "--expected",
            "",
        ],
    )
    assert result.exit_code == 1
    assert "Invalid JSON" in result.stdout


def test_cli_improve_uses_custom_improvement_model(monkeypatch, tmp_path) -> None:
    pipe = tmp_path / "pipe.py"
    pipe.write_text(
        "from flujo.domain import Step\n"
        "from flujo.testing.utils import StubAgent\n"
        "pipeline = Step.solution(StubAgent(['a']))\n"
    )
    data = tmp_path / "data.py"
    data.write_text(
        "from pydantic_evals import Dataset, Case\ndataset=Dataset(cases=[Case(inputs='a')])"
    )

    called: dict[str, Any] = {}

    def fake_make(model: str | None = None):
        called["model"] = model

        class A:
            async def run(self, p):
                return "{}"

        return A()

    monkeypatch.setattr("flujo.cli.main.make_self_improvement_agent", fake_make)
    from flujo.domain.models import ImprovementReport

    monkeypatch.setattr(
        "flujo.cli.main.evaluate_and_improve",
        AsyncMock(return_value=ImprovementReport(suggestions=[])),
    )

    result = runner.invoke(
        app,
        ["improve", str(pipe), str(data), "--improvement-model", "custom"],
    )
    assert result.exit_code == 0
    assert called["model"] == "custom"
