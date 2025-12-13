from typing import Any, ClassVar

import builtins

import pytest


class FakeConsole:
    """Minimal Console stub that captures printed strings.

    display_pipeline_results imports Console inside the function from rich.console,
    so tests patch rich.console.Console to this class.
    """

    captured: ClassVar[list[str]] = []

    def __init__(self, *args: Any, **_kwargs: Any) -> None:
        pass

    def print(self, *args: Any, **kwargs: Any) -> None:
        # Coerce to strings; rich objects can remain as their reprs
        text = " ".join([builtins.str(a) for a in args])
        FakeConsole.captured.append(text)


class DummyStep:
    def __init__(self) -> None:
        self.name = "final"
        self.output = "ok"
        self.success = True
        self.cost_usd = 0.0
        self.token_counts = 1
        self.latency_s = 0.0
        self.step_history: list[Any] = []


class DummyContext:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.status = "completed"

    # Match the interface used by display_pipeline_results when printing final context
    def model_dump(self) -> dict:
        return {"run_id": self.run_id, "status": self.status}


class DummyResult:
    def __init__(self, run_id: str) -> None:
        self.success = True
        self.step_history = [DummyStep()]
        self.total_cost_usd = 0.0
        self.final_pipeline_context = DummyContext(run_id)


@pytest.fixture(autouse=True)
def clear_fake_console() -> None:
    FakeConsole.captured.clear()


def test_display_uses_generated_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch Console class used inside display_pipeline_results
    import rich.console as rich_console

    monkeypatch.setattr(rich_console, "Console", FakeConsole)

    from flujo.cli.helpers import display_pipeline_results

    res = DummyResult(run_id="generated_123")
    display_pipeline_results(res, run_id=None, json_output=False)

    text = "\n".join(FakeConsole.captured)
    assert "Run ID:" in text
    assert "generated_123" in text


def test_display_prefers_cli_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    import rich.console as rich_console

    monkeypatch.setattr(rich_console, "Console", FakeConsole)

    from flujo.cli.helpers import display_pipeline_results

    res = DummyResult(run_id="generated_123")
    display_pipeline_results(res, run_id="cli_provided", json_output=False)

    text = "\n".join(FakeConsole.captured)
    # Ensure CLI-provided run_id is shown instead of the generated one
    assert "Run ID:" in text
    assert "cli_provided" in text
    assert "generated_123" not in text.split("Run ID:")[-1]
