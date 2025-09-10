from typing import Any, AsyncIterator

import pytest
from flujo.application.runner import Flujo
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline

# Ensure inclusion in fast suite
pytestmark = pytest.mark.fast


async def _echo(x: Any) -> Any:
    return x


def _make_single_step_pipeline(name: str = "s1") -> Pipeline[Any, Any]:
    step = Step.from_callable(_echo, name=name)
    return Pipeline(steps=[step])


def _make_two_step_pipeline() -> Pipeline[Any, Any]:
    s1 = Step.from_callable(_echo, name="first")
    s2 = Step.from_callable(_echo, name="second")
    return Pipeline(steps=[s1, s2])


@pytest.mark.asyncio
async def test_missing_terminal_outcome_synthesizes_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a step produces no terminal outcome, ExecutionManager synthesizes a Failure.

    This guards against the bug where a step would "disappear" and the pipeline
    incorrectly reported completion without a StepResult.
    """

    # 1) Build a minimal 1-step pipeline
    p = _make_single_step_pipeline()

    # 2) Patch StepCoordinator.execute_step to yield nothing (no terminal outcome)
    from flujo.application.core import step_coordinator as sc_mod

    original_execute_step = sc_mod.StepCoordinator.execute_step

    async def _no_outcome_execute_step(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        if False:
            yield None  # make this an async generator but yield no items
        return

    monkeypatch.setattr(sc_mod.StepCoordinator, "execute_step", _no_outcome_execute_step)

    # 3) Also patch backend to return None on the manager's direct retry path so we hit the final synthesizer
    from flujo.infra.backends import LocalBackend

    original_backend_exec = LocalBackend.execute_step

    async def _backend_none(self, *args: Any, **kwargs: Any) -> Any:
        return None

    monkeypatch.setattr(LocalBackend, "execute_step", _backend_none)

    try:
        runner: Flujo[Any, Any, Any] = Flujo(pipeline=p)
        result = runner.run("")
    finally:
        # Restore (pytest monkeypatch will also undo, but keep explicit for clarity)
        monkeypatch.setattr(sc_mod.StepCoordinator, "execute_step", original_execute_step)
        monkeypatch.setattr(LocalBackend, "execute_step", original_backend_exec)

    # 4) Assertions: one synthesized failure is recorded and overall success is False
    assert len(result.step_history) == 1
    sr = result.step_history[0]
    assert sr.success is False
    assert isinstance(sr.feedback, str) and "no terminal outcome" in sr.feedback.lower()
    assert result.success is False


@pytest.mark.asyncio
async def test_completion_gate_requires_all_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runner marks pipeline completed only when all steps succeeded and are present.

    We drop the second step's outcome and ensure the run is not marked as success.
    """

    p = _make_two_step_pipeline()

    # Let the first call behave normally; second call yields nothing
    from flujo.application.core import step_coordinator as sc_mod

    call_count = {"n": 0}

    original_execute_step = sc_mod.StepCoordinator.execute_step

    async def _first_ok_second_missing(
        self,
        step,
        data,
        context,
        backend=None,
        *,
        stream=False,
        step_executor=None,
        usage_limits=None,
        quota=None,
    ):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Delegate to original for the first step
            async for item in original_execute_step(
                self,
                step,
                data,
                context,
                backend=backend,
                stream=stream,
                step_executor=step_executor,
                usage_limits=usage_limits,
                quota=quota,
            ):
                yield item
            return
        # Second step: emit no terminal outcome
        if False:
            yield None
        return

    # Patch coordinator and backend retry to None to force synthesized failure
    monkeypatch.setattr(sc_mod.StepCoordinator, "execute_step", _first_ok_second_missing)

    from flujo.infra.backends import LocalBackend

    async def _backend_none(self, *args: Any, **kwargs: Any) -> Any:
        return None

    monkeypatch.setattr(LocalBackend, "execute_step", _backend_none)

    runner: Flujo[Any, Any, Any] = Flujo(pipeline=p)
    result = runner.run("")

    # We should have exactly 2 steps in the history (first success, second synthesized failure)
    assert len(result.step_history) == 2
    assert result.step_history[0].success is True
    assert result.step_history[1].success is False
    # Overall pipeline must not be marked success
    assert result.success is False


def test_structured_output_single_string_autowrap() -> None:
    """Structured finalize path auto-wraps a plain string into a one-field object schema."""

    async def _just_text(_: Any) -> str:
        return "hello world"

    st = Step.from_callable(_just_text, name="wrapme")
    # Declare structured_output with a single required string field
    st.meta.setdefault("processing", {})
    st.meta["processing"].update(
        {
            "structured_output": "openai_json",
            "schema": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        }
    )
    p = Pipeline(steps=[st])
    runner: Flujo[Any, Any, Any] = Flujo(pipeline=p)
    result = runner.run("")
    assert len(result.step_history) == 1
    sr = result.step_history[0]
    assert sr.success is True
    assert isinstance(sr.output, dict)
    assert sr.output.get("summary") == "hello world"
