from typing import Any

# Tests require an API key; a fixture sets it for each test

from flujo.cli.main import app
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import json
from flujo.infra.settings import Settings

runner = CliRunner()


# Helper dummy agent
class DummyAgent:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        return "mocked agent output"

    async def run_async(self, *args, **kwargs):
        return "mocked agent output"

    def model_dump(self):
        return {"solution": "mocked", "score": 1.0}


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch) -> None:
    """Ensure OPENAI_API_KEY is present and refresh settings for each test."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import sys

    new_settings = Settings()
    settings_module = sys.modules["flujo.infra.settings"]
    monkeypatch.setattr(settings_module, "settings", new_settings)


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

    async def dummy_run_async(pipeline, task):
        return DummyCandidate()

    monkeypatch.setattr("flujo.cli.main.run_default_pipeline", dummy_run_async)
    monkeypatch.setattr("flujo.cli.main.make_review_agent", lambda *a, **k: DummyAgent())
    monkeypatch.setattr("flujo.cli.main.make_solution_agent", lambda *a, **k: DummyAgent())
    monkeypatch.setattr("flujo.cli.main.make_validator_agent", lambda *a, **k: DummyAgent())

    result = runner.invoke(app, ["dev", "experimental", "solve", "write a poem"])
    assert result.exit_code == 0
    assert '"solution": "mocked"' in result.stdout


def test_cli_solve_custom_models(monkeypatch) -> None:
    class DummyCandidate:
        def model_dump(self):
            return {"solution": "mocked", "score": 1.0}

    async def dummy_run_async(pipeline, task):
        return DummyCandidate()

    # Patch the correct agent factories
    monkeypatch.setattr("flujo.cli.main.run_default_pipeline", dummy_run_async)
    monkeypatch.setattr("flujo.cli.main.make_review_agent", lambda *a, **k: DummyAgent())
    monkeypatch.setattr("flujo.cli.main.make_solution_agent", lambda *a, **k: DummyAgent())
    monkeypatch.setattr("flujo.cli.main.make_validator_agent", lambda *a, **k: DummyAgent())
    result = runner.invoke(
        app, ["dev", "experimental", "solve", "write", "--solution-model", "gemini:gemini-1.5-pro"]
    )
    assert result.exit_code == 0


def test_cli_bench_command(monkeypatch) -> None:
    pytest.importorskip("numpy")

    class DummyCandidate:
        score = 1.0

        def model_dump(self):
            return {"solution": "mocked", "score": 1.0}

    async def dummy_run_async(pipeline, task):
        return DummyCandidate()

    monkeypatch.setattr("flujo.cli.main.run_default_pipeline", dummy_run_async)
    monkeypatch.setattr("flujo.cli.main.make_review_agent", lambda *a, **k: DummyAgent())
    monkeypatch.setattr("flujo.cli.main.make_solution_agent", lambda *a, **k: DummyAgent())
    monkeypatch.setattr("flujo.cli.main.make_validator_agent", lambda *a, **k: DummyAgent())

    result = runner.invoke(app, ["dev", "experimental", "bench", "test prompt", "--rounds", "2"])
    assert result.exit_code == 0
    assert "Benchmark Results" in result.stdout


def test_cli_solve_with_weights(monkeypatch) -> None:
    from flujo.domain.models import Task

    class DummyCandidate:
        score = 1.0

        def model_dump(self):
            return {"solution": "mocked", "score": self.score}

    mock_agent = AsyncMock()
    mock_agent.run.return_value = "mocked agent output"
    mock_agent.run_async.return_value = "mocked agent output"

    async def mock_run_async(pipeline, task: Task) -> DummyCandidate:
        assert isinstance(task, Task)
        assert task.prompt == "write a poem"
        assert task.metadata.get("weights") is not None
        return DummyCandidate()

    monkeypatch.setattr("flujo.cli.main.run_default_pipeline", mock_run_async)
    monkeypatch.setattr("flujo.cli.main.make_review_agent", lambda *a, **k: mock_agent)
    monkeypatch.setattr("flujo.cli.main.make_solution_agent", lambda *a, **k: mock_agent)
    monkeypatch.setattr("flujo.cli.main.make_validator_agent", lambda *a, **k: mock_agent)

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

        result = runner.invoke(
            app, ["dev", "experimental", "solve", "write a poem", "--weights-path", weights_file]
        )

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

    finally:
        if weights_file and os.path.exists(weights_file):
            os.remove(weights_file)


def test_cli_solve_weights_file_not_found() -> None:
    result = runner.invoke(
        app, ["dev", "experimental", "solve", "prompt", "--weights-path", "nonexistent.json"]
    )
    assert result.exit_code == 1
    assert "Weights file not found" in result.stderr


def test_cli_solve_weights_file_invalid_json(tmp_path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not a json")
    result = runner.invoke(
        app, ["dev", "experimental", "solve", "prompt", "--weights-path", str(bad_file)]
    )
    assert result.exit_code == 1
    assert "Error" in result.stdout or "Traceback" in result.stdout or result.stderr


def test_cli_solve_weights_invalid_structure(tmp_path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text('{"item": "a", "weight": 1}')
    result = runner.invoke(
        app, ["dev", "experimental", "solve", "prompt", "--weights-path", str(bad_file)]
    )
    assert result.exit_code == 1
    assert "list of objects" in result.stderr


def test_cli_solve_weights_missing_keys(tmp_path) -> None:
    weights = [{"item": "a"}]
    file = tmp_path / "weights.json"
    file.write_text(json.dumps(weights))
    result = runner.invoke(
        app, ["dev", "experimental", "solve", "prompt", "--weights-path", str(file)]
    )
    assert result.exit_code == 1
    assert "list of objects" in result.stderr


def test_cli_solve_context_data_safe_deserialize(tmp_path, monkeypatch) -> None:
    calls = {}

    def fake_deserialize(data):
        calls["called"] = True
        return data

    pipeline = tmp_path / "pipe.py"
    pipeline.write_text(
        "from flujo.domain import Step\nfrom flujo.testing.utils import StubAgent\n"
        "s1 = Step.model_validate({'name': 'A', 'agent': StubAgent(['x'])})\n"
        "s1.__step_output_type__ = str\n"
        "s2 = Step.model_validate({'name': 'B', 'agent': StubAgent(['y'])})\n"
        "s2.__step_input_type__ = str\n"
        "s2.meta = {'is_adapter': True, 'adapter_id': 'generic-adapter', 'adapter_allow': 'generic'}\n"
        "pipeline = s1 >> s2\n"
    )

    monkeypatch.setattr("flujo.cli.main.safe_deserialize", fake_deserialize)

    from flujo.domain.models import PipelineResult

    monkeypatch.setattr("flujo.cli.main.Flujo.run", lambda self, inp: PipelineResult())

    result = runner.invoke(
        app,
        ["run", str(pipeline), "--input", "hi", "--context-data", '{"a":1}', "--json"],
    )
    assert result.exit_code == 0
    assert calls.get("called")


def test_cli_add_eval_case_uses_safe_deserialize(tmp_path, monkeypatch) -> None:
    file = tmp_path / "data.py"
    file.write_text("dataset = None")

    calls = {}

    def fake_deserialize(data):
        calls["called"] = True
        return data

    monkeypatch.setattr("flujo.cli.dev_commands.safe_deserialize", fake_deserialize)

    result = runner.invoke(
        app,
        [
            "dev",
            "experimental",
            "add-case",
            "-d",
            str(file),
            "-n",
            "case",
            "-i",
            "x",
            "--metadata",
            '{"num":1}',
            "--expected",
            "",
        ],
    )
    assert result.exit_code == 0
    assert calls.get("called")


def test_cli_solve_weights_file_safe_deserialize(tmp_path, monkeypatch) -> None:
    weights = [{"item": "a", "weight": 1.0}]
    weights_file = tmp_path / "w.json"
    weights_file.write_text(json.dumps(weights))

    calls = {}

    def fake_deserialize(data):
        calls["called"] = True
        return data

    monkeypatch.setattr("flujo.cli.main.safe_deserialize", fake_deserialize)

    class DummyCandidate:
        def model_dump(self):
            return {}

    async def dummy_run_async(pipeline, task):
        return DummyCandidate()

    monkeypatch.setattr("flujo.cli.main.run_default_pipeline", dummy_run_async)
    monkeypatch.setattr("flujo.cli.main.make_review_agent", lambda *a, **k: DummyAgent())
    monkeypatch.setattr("flujo.cli.main.make_solution_agent", lambda *a, **k: DummyAgent())
    monkeypatch.setattr("flujo.cli.main.make_validator_agent", lambda *a, **k: DummyAgent())

    # from flujo.cli.main import app # This line is moved to the top of the file

    result = runner.invoke(
        app, ["dev", "experimental", "solve", "prompt", "--weights-path", str(weights_file)]
    )
    assert result.exit_code == 0
    assert calls["called"]


def test_cli_solve_keyboard_interrupt(monkeypatch) -> None:
    def raise_keyboard(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr("flujo.cli.main.run_default_pipeline", raise_keyboard)
    monkeypatch.setattr("flujo.cli.main.make_review_agent", lambda *a, **k: DummyAgent())
    monkeypatch.setattr("flujo.cli.main.make_solution_agent", lambda *a, **k: DummyAgent())
    monkeypatch.setattr("flujo.cli.main.make_validator_agent", lambda *a, **k: DummyAgent())

    # from flujo.cli.main import app # This line is moved to the top of the file

    result = runner.invoke(app, ["dev", "experimental", "solve", "write a poem"])
    assert result.exit_code == 130


def test_cli_bench_keyboard_interrupt(monkeypatch) -> None:
    pytest.importorskip("numpy")

    def raise_keyboard(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr("flujo.cli.main.run_default_pipeline", raise_keyboard)
    monkeypatch.setattr("flujo.cli.main.make_review_agent", lambda *a, **k: DummyAgent())
    monkeypatch.setattr("flujo.cli.main.make_solution_agent", lambda *a, **k: DummyAgent())
    monkeypatch.setattr("flujo.cli.main.make_validator_agent", lambda *a, **k: DummyAgent())

    # from flujo.cli.main import app # This line is moved to the top of the file

    result = runner.invoke(app, ["dev", "experimental", "bench", "test prompt", "--rounds", "2"])
    assert result.exit_code == 130


def test_cli_version_cmd_package_not_found(monkeypatch) -> None:
    import importlib.metadata

    def raise_package_not_found(_name):
        raise importlib.metadata.PackageNotFoundError("fail")

    monkeypatch.setattr(
        importlib.metadata,
        "version",
        raise_package_not_found,
    )
    # from flujo.cli.main import app # This line is moved to the top of the file

    result = runner.invoke(app, ["dev", "version"])
    assert result.exit_code == 0
    assert "unknown" in result.stdout


def test_cli_main_callback_profile(monkeypatch) -> None:
    # Should not raise, just configure logfire
    result = runner.invoke(app, ["--profile"])
    assert result.exit_code == 0 or result.exit_code == 2


def test_cli_solve_configuration_error(monkeypatch) -> None:
    """Test that configuration errors surface with exit code 2."""

    from flujo.exceptions import ConfigurationError

    def raise_config_error(*args, **kwargs):
        raise ConfigurationError("Missing API key!")

    # Patch all agent factories to raise ConfigurationError
    monkeypatch.setattr("flujo.cli.main.make_review_agent", raise_config_error)
    monkeypatch.setattr("flujo.cli.main.make_solution_agent", raise_config_error)
    monkeypatch.setattr("flujo.cli.main.make_validator_agent", raise_config_error)

    result = runner.invoke(app, ["dev", "experimental", "solve", "prompt"])
    assert result.exit_code == 2
    assert "Configuration Error: Missing API key!" in result.stderr


def test_cli_explain(tmp_path) -> None:
    file = tmp_path / "pipe.py"
    file.write_text(
        "from flujo.domain import Step\npipeline = Step.model_validate({'name': 'A'}) >> Step.model_validate({'name': 'B'})\n"
    )

    result = runner.invoke(app, ["dev", "explain", str(file)])
    assert result.exit_code == 0
    assert "A" in result.stdout
    assert "B" in result.stdout


def test_cli_validate_success(tmp_path) -> None:
    file = tmp_path / "pipe.py"
    file.write_text(
        "from flujo.domain import Step\nfrom flujo.testing.utils import StubAgent\n"
        "s1 = Step.model_validate({'name': 'A', 'agent': StubAgent(['x'])})\n"
        "s1.__step_output_type__ = str\n"
        "s2 = Step.model_validate({'name': 'B', 'agent': StubAgent(['y']), 'meta': {'is_adapter': True, 'adapter_id': 'generic-adapter', 'adapter_allow': 'generic'}})\n"
        "s2.__step_input_type__ = str\n"
        "pipeline = s1 >> s2\n"
    )
    result = runner.invoke(app, ["validate", str(file)])
    assert result.exit_code == 0
    assert "Pipeline is valid" in result.stdout


def test_cli_validate_failure(tmp_path) -> None:
    file = tmp_path / "pipe.py"
    file.write_text(
        "from flujo.domain import Step\npipeline = Step.model_validate({'name': 'A'}) >> Step.model_validate({'name': 'B'})\n"
    )
    result = runner.invoke(app, ["validate", str(file), "--strict"])
    # Strict validation now uses stable exit code 4 (EX_VALIDATION_FAILED)
    assert result.exit_code == 4


def test_cli_improve_output_formatting(monkeypatch, tmp_path) -> None:
    pipe = tmp_path / "pipe.py"
    pipe.write_text(
        "from flujo.domain import Step\n"
        "from flujo.testing.utils import StubAgent\n"
        "pipeline = Step.model_validate({'name': 'A', 'agent': StubAgent(['a'])})\n"
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

    result = runner.invoke(app, ["dev", "experimental", "improve", str(pipe), str(data)])
    assert result.exit_code == 0
    assert "IMPROVEMENT REPORT" in result.stdout


def test_cli_improve_json_output(monkeypatch, tmp_path) -> None:
    pipe = tmp_path / "pipe.py"
    pipe.write_text(
        "from flujo.domain import Step\n"
        "from flujo.testing.utils import StubAgent\n"
        "pipeline = Step.model_validate({'name': 'A', 'agent': StubAgent(['a'])})\n"
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

    result = runner.invoke(app, ["dev", "experimental", "improve", str(pipe), str(data), "--json"])
    assert result.exit_code == 0
    assert '"suggestions"' in result.stdout


def test_cli_help() -> None:
    """Test that the help command works and shows all available commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Top-level is focused for YAML users; advanced commands live under 'dev'
    assert "dev" in result.stdout
    assert "validate" in result.stdout


def test_cli_run() -> None:
    """Test basic run functionality with default settings."""
    from unittest.mock import patch

    class DummyCandidate:
        def model_dump(self):
            return {"solution": "test solution", "score": 1.0}

    async def dummy_run_async(pipeline, task):
        return DummyCandidate()

    with patch("flujo.cli.main.run_default_pipeline", dummy_run_async):
        # from flujo.cli.main import app # This line is moved to the top of the file

        result = runner.invoke(app, ["dev", "experimental", "solve", "test prompt"])
        assert result.exit_code == 0


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

    async def dummy_run_async(pipeline, task):
        return DummyCandidate()

    with (
        patch("flujo.cli.main.run_default_pipeline", dummy_run_async),
        patch("flujo.cli.main.make_review_agent", return_value=DummyAgent()),
        patch("flujo.cli.main.make_solution_agent", return_value=DummyAgent()),
        patch("flujo.cli.main.make_validator_agent", return_value=DummyAgent()),
        patch("flujo.cli.main.get_reflection_agent"),
        patch("flujo.cli.main.load_settings", return_value=dummy_settings),
    ):
        # from flujo.cli.main import app # This line is moved to the top of the file

        result = runner.invoke(
            app,
            [
                "dev",
                "experimental",
                "solve",
                "test prompt",
                "--solution-model",
                "gpt-4",
                "--review-model",
                "gpt-3.5-turbo",
                "--validator-model",
                "gpt-3.5-turbo",
                "--reflection-model",
                "gpt-4",
                "--reflection",
                "--scorer",
                "ratio",
            ],
        )
        assert result.exit_code == 0


def test_cli_run_with_invalid_args() -> None:
    """Test run with invalid command line arguments."""
    # Test with invalid max-iters
    result = runner.invoke(
        app, ["dev", "experimental", "solve", "test prompt", "--max-iters", "-1"]
    )
    assert result.exit_code != 0
    assert "Error" in result.stderr

    # Test with invalid k
    result = runner.invoke(app, ["dev", "experimental", "solve", "test prompt", "--k", "0"])
    assert result.exit_code != 0
    assert "Error" in result.stderr

    # Test with invalid scorer
    result = runner.invoke(
        app, ["dev", "experimental", "solve", "test prompt", "--scorer", "invalid"]
    )
    assert result.exit_code != 0
    assert "Error" in result.stderr


def test_cli_run_with_invalid_model() -> None:
    """Test run with invalid model names."""
    from unittest.mock import patch
    from flujo.exceptions import ConfigurationError

    with patch("flujo.cli.main.make_review_agent") as mock_review:
        mock_review.side_effect = ConfigurationError("Invalid model name")
        result = runner.invoke(
            app,
            ["dev", "experimental", "solve", "test prompt", "--solution-model", "invalid-model"],
        )
        assert result.exit_code == 2
        assert "Configuration Error" in result.stderr


def test_cli_run_with_invalid_retries() -> None:
    """Test run with invalid retry settings."""
    from flujo.exceptions import ConfigurationError

    def raise_config_error(*args, **kwargs):
        raise ConfigurationError("Invalid retry settings")

    with patch("flujo.cli.main.run_default_pipeline", raise_config_error):
        # from flujo.cli.main import app # This line is moved to the top of the file

        result = runner.invoke(app, ["dev", "experimental", "solve", "test prompt"])
        assert result.exit_code == 2


def test_cli_run_with_invalid_agent_timeout() -> None:
    """Test run with invalid agent timeout settings."""
    from flujo.exceptions import ConfigurationError

    def raise_config_error(*args, **kwargs):
        raise ConfigurationError("Invalid agent timeout settings")

    with patch("flujo.cli.main.run_default_pipeline", raise_config_error):
        # from flujo.cli.main import app # This line is moved to the top of the file

        result = runner.invoke(app, ["dev", "experimental", "solve", "test prompt"])
        assert result.exit_code == 2


def test_cli_run_with_invalid_review_model() -> None:
    """Test run with invalid review model."""
    from unittest.mock import patch
    from flujo.exceptions import ConfigurationError

    with patch("flujo.cli.main.make_review_agent") as mock_review:
        mock_review.side_effect = ConfigurationError("Invalid review model")
        result = runner.invoke(
            app, ["dev", "experimental", "solve", "test prompt", "--review-model", "invalid-model"]
        )
        assert result.exit_code == 2
        assert "Configuration Error" in result.stderr


def test_cli_run_with_invalid_review_model_path() -> None:
    """Test run with invalid review model path."""
    from unittest.mock import patch
    from flujo.exceptions import ConfigurationError

    with patch("flujo.cli.main.make_review_agent") as mock_review:
        mock_review.side_effect = ConfigurationError("Invalid review model path")
        result = runner.invoke(
            app, ["dev", "experimental", "solve", "test prompt", "--review-model", "/invalid/path"]
        )
        assert result.exit_code == 2
        assert "Configuration Error" in result.stderr


def test_cli_add_eval_case_prints_correct_case_string(tmp_path) -> None:
    file = tmp_path / "data.py"
    file.write_text("dataset = None")
    result = runner.invoke(
        app,
        [
            "dev",
            "experimental",
            "add-case",
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
        [
            "dev",
            "experimental",
            "add-case",
            "-d",
            str(missing),
            "-n",
            "a",
            "-i",
            "b",
            "--expected",
            "",
        ],
    )
    assert result.exit_code == 1
    assert "Dataset file not found" in result.stdout


def test_cli_add_eval_case_invalid_metadata_json(tmp_path) -> None:
    file = tmp_path / "data.py"
    file.write_text("dataset = None")
    result = runner.invoke(
        app,
        [
            "dev",
            "experimental",
            "add-case",
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
        "pipeline = Step.model_validate({'name': 'A', 'agent': StubAgent(['a'])})\n"
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
        ["dev", "experimental", "improve", str(pipe), str(data), "--improvement-model", "custom"],
    )
    assert result.exit_code == 0
    assert called["model"] == "custom"


def test_apply_cli_defaults_helper(monkeypatch):
    """Test the apply_cli_defaults helper function."""
    from flujo.cli.main import apply_cli_defaults

    # Mock the get_cli_defaults to return no defaults for the first test
    def mock_get_defaults_empty(command):
        return {}

    monkeypatch.setattr("flujo.cli.main.get_cli_defaults", mock_get_defaults_empty)

    # Test with no defaults (should return original values)
    result = apply_cli_defaults("solve", max_iters=None, k=5)
    assert result["max_iters"] is None
    assert result["k"] == 5

    # Test with defaults applied
    # Mock the get_cli_defaults to return some defaults
    def mock_get_defaults(command):
        if command == "solve":
            return {"max_iters": 10, "k": 3}
        return {}

    monkeypatch.setattr("flujo.cli.main.get_cli_defaults", mock_get_defaults)

    result = apply_cli_defaults("solve", max_iters=None, k=None)
    assert result["max_iters"] == 10
    assert result["k"] == 3

    # Test that provided values are not overridden
    result = apply_cli_defaults("solve", max_iters=5, k=None)
    assert result["max_iters"] == 5  # Should not be overridden
    assert result["k"] == 3  # Should be overridden
