import importlib
import pytest
from typer.testing import CliRunner
from flujo import Flujo, Step
from flujo.cli.main import app
from flujo.testing.utils import StubAgent, gather_result


def test_core_imports() -> None:
    assert importlib.import_module("flujo")
    assert Flujo
    assert Step


@pytest.mark.asyncio
async def test_basic_pipeline_runs() -> None:
    step = Step.model_validate({"name": "s1", "agent": StubAgent(["ok"])})
    result = await gather_result(Flujo(step), "hi")
    assert result.step_history[-1].output == "ok"


def test_cli_version() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["version-cmd"])
    assert res.exit_code == 0
    assert "flujo version:" in res.stdout
