from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from flujo.domain.dsl import Step, Pipeline
from flujo.application.runner import Flujo
from flujo.domain.models import UsageLimits
from flujo.domain.events import PreRunPayload


def _make_echo_pipeline() -> Pipeline[str, str]:
    async def echo(x: str) -> str:
        return x

    return Pipeline.from_step(Step.from_callable(echo, name="echo"))


def _write_basic_budgets_toml(tmp_path: Path) -> None:
    (tmp_path / "flujo.toml").write_text(
        (
            """
            [budgets.default]
            total_cost_usd_limit = 10.0
            total_tokens_limit = 1000

            [budgets.pipeline]
            "demo" = { total_cost_usd_limit = 3.5, total_tokens_limit = 77 }
            "team-*" = { total_cost_usd_limit = 5.0 }
            """
        ).strip()
    )


def test_runner_budget_resolution_exact_match_and_precedence(tmp_path: Path, monkeypatch) -> None:
    _write_basic_budgets_toml(tmp_path)
    monkeypatch.chdir(tmp_path)

    # Pipeline and code-provided limits (looser than TOML to test min rule)
    pipeline = _make_echo_pipeline()
    code_limits = UsageLimits(total_cost_usd_limit=8.0, total_tokens_limit=5000)

    # Capture pre_run budget values via hook
    seen: List[Tuple[Optional[float], Optional[int]]] = []

    async def capture(payload: PreRunPayload) -> None:
        seen.append((payload.initial_budget_cost_usd, payload.initial_budget_tokens))

    runner = Flujo(
        pipeline=pipeline, pipeline_name="demo", usage_limits=code_limits, hooks=[capture]
    )
    result = runner.run("hello")
    assert result is not None

    # Effective limits should be min(code, toml_exact): (3.5, 77)
    assert runner.usage_limits is not None
    assert runner.usage_limits.total_cost_usd_limit == 3.5
    assert runner.usage_limits.total_tokens_limit == 77

    # Hook observed values should match effective limits
    assert seen, "pre_run hook did not fire"
    cost, tokens = seen[0]
    assert cost == 3.5
    assert tokens == 77


def test_runner_budget_resolution_wildcard(tmp_path: Path, monkeypatch) -> None:
    _write_basic_budgets_toml(tmp_path)
    monkeypatch.chdir(tmp_path)

    pipeline = _make_echo_pipeline()

    # No code-provided limits; wildcard should apply (cost=5.0, tokens=None)
    seen: List[Tuple[Optional[float], Optional[int]]] = []

    async def capture(payload: PreRunPayload) -> None:
        seen.append((payload.initial_budget_cost_usd, payload.initial_budget_tokens))

    runner = Flujo(pipeline=pipeline, pipeline_name="team-alpha", hooks=[capture])
    _ = runner.run("hello")

    assert runner.usage_limits is not None
    assert runner.usage_limits.total_cost_usd_limit == 5.0
    assert runner.usage_limits.total_tokens_limit is None

    assert seen, "pre_run hook did not fire"
    cost, tokens = seen[0]
    assert cost == 5.0
    assert tokens is None


def test_runner_budget_resolution_default(tmp_path: Path, monkeypatch) -> None:
    _write_basic_budgets_toml(tmp_path)
    monkeypatch.chdir(tmp_path)

    pipeline = _make_echo_pipeline()

    # Unknown pipeline name falls back to default: (10.0, 1000)
    seen: List[Tuple[Optional[float], Optional[int]]] = []

    async def capture(payload: PreRunPayload) -> None:
        seen.append((payload.initial_budget_cost_usd, payload.initial_budget_tokens))

    runner = Flujo(pipeline=pipeline, pipeline_name="unknown", hooks=[capture])
    _ = runner.run("hello")

    assert runner.usage_limits is not None
    assert runner.usage_limits.total_cost_usd_limit == 10.0
    assert runner.usage_limits.total_tokens_limit == 1000

    assert seen, "pre_run hook did not fire"
    cost, tokens = seen[0]
    assert cost == 10.0
    assert tokens == 1000
