import pytest
from typing import Any
from pydantic import BaseModel

from flujo.domain import Step, Pipeline
from flujo.testing.utils import StubAgent, DummyPlugin, gather_result
from flujo.domain.plugins import PluginOutcome
from tests.conftest import create_test_flujo


class EchoAgent:
    async def run(self, data, **kwargs):
        return data


@pytest.mark.asyncio
async def test_branch_a_executes() -> None:
    classify = Step.model_validate({"name": "classify", "agent": StubAgent(["a"])})
    branches = {
        "a": Pipeline.from_step(Step.model_validate({"name": "a", "agent": StubAgent(["A"])})),
        "b": Pipeline.from_step(Step.model_validate({"name": "b", "agent": StubAgent(["B"])})),
    }
    branch_step = Step.branch_on(
        name="branch",
        condition_callable=lambda out, ctx: out,
        branches=branches,
    )
    runner = create_test_flujo(classify >> branch_step)
    result = await gather_result(runner, "in")
    step_result = result.step_history[-1]
    assert step_result.success is True
    assert step_result.output == "A"
    assert step_result.metadata_["executed_branch_key"] == "a"


@pytest.mark.asyncio
async def test_branch_b_executes() -> None:
    classify = Step.model_validate({"name": "classify", "agent": StubAgent(["b"])})
    branches = {
        "a": Pipeline.from_step(Step.model_validate({"name": "a", "agent": StubAgent(["A"])})),
        "b": Pipeline.from_step(Step.model_validate({"name": "b", "agent": StubAgent(["B"])})),
    }
    branch_step = Step.branch_on(
        name="branch",
        condition_callable=lambda out, ctx: out,
        branches=branches,
    )
    runner = create_test_flujo(classify >> branch_step)
    result = await gather_result(runner, "in")
    assert result.step_history[-1].output == "B"
    assert result.step_history[-1].metadata_["executed_branch_key"] == "b"


@pytest.mark.asyncio
async def test_default_branch_used() -> None:
    classify = Step.model_validate({"name": "classify", "agent": StubAgent(["x"])})
    branches = {
        "a": Pipeline.from_step(Step.model_validate({"name": "a", "agent": StubAgent(["A"])})),
    }
    default = Pipeline.from_step(Step.model_validate({"name": "def", "agent": StubAgent(["DEF"])}))
    branch_step = Step.branch_on(
        name="branch",
        condition_callable=lambda out, ctx: out,
        branches=branches,
        default_branch_pipeline=default,
    )
    runner = create_test_flujo(classify >> branch_step)
    result = await gather_result(runner, "in")
    assert result.step_history[-1].output == "DEF"
    assert result.step_history[-1].metadata_["executed_branch_key"] == "x"


@pytest.mark.asyncio
async def test_no_match_no_default_fails() -> None:
    classify = Step.model_validate({"name": "classify", "agent": StubAgent(["x"])})
    branches = {
        "a": Pipeline.from_step(Step.model_validate({"name": "a", "agent": StubAgent(["A"])}))
    }
    branch_step = Step.branch_on(
        name="branch",
        condition_callable=lambda out, ctx: out,
        branches=branches,
    )
    runner = create_test_flujo(classify >> branch_step)
    result = await gather_result(runner, "in")
    step_result = result.step_history[-1]
    assert step_result.success is False
    assert "no default" in step_result.feedback.lower()


class FlagCtx(BaseModel):
    flag: str = "a"


@pytest.mark.asyncio
async def test_condition_uses_context() -> None:
    classify = Step.model_validate({"name": "classify", "agent": StubAgent(["ignored"])})
    branches = {
        "a": Pipeline.from_step(Step.model_validate({"name": "a", "agent": StubAgent(["A"])})),
        "b": Pipeline.from_step(Step.model_validate({"name": "b", "agent": StubAgent(["B"])})),
    }
    branch_step = Step.branch_on(
        name="branch",
        condition_callable=lambda out, ctx: ctx.flag if ctx else "a",
        branches=branches,
    )
    runner = create_test_flujo(
        classify >> branch_step,
        context_model=FlagCtx,
        initial_context_data={"flag": "b"},
    )
    result = await gather_result(runner, "in")
    assert result.step_history[-1].output == "B"
    assert result.step_history[-1].metadata_["executed_branch_key"] == "b"


@pytest.mark.asyncio
async def test_mappers_applied() -> None:
    branches = {
        "x": Pipeline.from_step(Step.model_validate({"name": "inc", "agent": EchoAgent()})),
    }
    branch_step = Step.branch_on(
        name="branch",
        condition_callable=lambda out, ctx: "x",
        branches=branches,
        branch_input_mapper=lambda inp, ctx: inp + 1,
        branch_output_mapper=lambda out, key, ctx: out * 10,
    )
    runner = create_test_flujo(branch_step)
    result = await gather_result(runner, 1)
    assert result.step_history[-1].output == 20


@pytest.mark.asyncio
async def test_failure_in_branch_propagates() -> None:
    fail_plugin = DummyPlugin([PluginOutcome(success=False, feedback="bad")])
    bad_step = Step.model_validate(
        {"name": "bad", "agent": StubAgent(["oops"]), "plugins": [(fail_plugin, 0)]}
    )
    branches = {"a": Pipeline.from_step(bad_step)}
    branch_step = Step.branch_on(
        name="branch",
        condition_callable=lambda out, ctx: "a",
        branches=branches,
    )
    runner = create_test_flujo(branch_step)
    result = await gather_result(runner, "in")
    step_result = result.step_history[-1]
    assert step_result.success is False
    assert "bad" in step_result.feedback


@pytest.mark.asyncio
async def test_condition_exception_fails_step() -> None:
    branches = {
        "a": Pipeline.from_step(Step.model_validate({"name": "a", "agent": StubAgent(["A"])}))
    }

    def condition(_: str, __: FlagCtx | None) -> str:
        raise RuntimeError("boom")

    branch_step = Step.branch_on(
        name="branch",
        condition_callable=condition,
        branches=branches,
    )
    runner = create_test_flujo(branch_step)
    result = await gather_result(runner, "in")
    step_result = result.step_history[-1]
    assert step_result.success is False
    assert "boom" in step_result.feedback


@pytest.mark.asyncio
async def test_conditional_step_error_in_condition_callable() -> None:
    branches = {
        "a": Pipeline.from_step(Step.model_validate({"name": "a", "agent": StubAgent(["A"])}))
    }

    def bad_condition(_: str, __: BaseModel | None) -> str:
        raise RuntimeError("cond err")

    branch_step = Step.branch_on(
        name="branch",
        condition_callable=bad_condition,
        branches=branches,
    )
    after = Step.model_validate({"name": "after", "agent": StubAgent(["after"])})
    runner = create_test_flujo(branch_step >> after)
    result = await gather_result(runner, "in")
    assert len(result.step_history) == 1
    step_result = result.step_history[0]
    assert step_result.success is False
    assert "cond err" in step_result.feedback


@pytest.mark.asyncio
async def test_conditional_step_error_in_branch_input_mapper() -> None:
    branches = {
        "a": Pipeline.from_step(Step.model_validate({"name": "a", "agent": StubAgent(["A"])}))
    }

    def branch_input(_: str, __: BaseModel | None) -> str:
        raise RuntimeError("input map err")

    branch_step = Step.branch_on(
        name="branch",
        condition_callable=lambda out, ctx: "a",
        branches=branches,
        branch_input_mapper=branch_input,
    )
    runner = create_test_flujo(branch_step)
    result = await gather_result(runner, "in")
    step_result = result.step_history[-1]
    assert step_result.success is False
    assert "input map err" in step_result.feedback


@pytest.mark.asyncio
async def test_conditional_step_error_in_branch_output_mapper() -> None:
    branches = {
        "a": Pipeline.from_step(Step.model_validate({"name": "a", "agent": StubAgent([1])}))
    }

    def out_map(_: int, __: str, ___: BaseModel | None) -> int:
        raise RuntimeError("output map err")

    branch_step = Step.branch_on(
        name="branch",
        condition_callable=lambda out, ctx: "a",
        branches=branches,
        branch_output_mapper=out_map,
    )
    runner = create_test_flujo(branch_step)
    result = await gather_result(runner, 0)
    step_result = result.step_history[-1]
    assert step_result.success is False
    assert "output map err" in step_result.feedback


@pytest.mark.asyncio
async def test_conditional_step_branch_input_mapper_flow() -> None:
    captured: list[tuple[int, BaseModel | None]] = []

    def inp_map(inp: int, ctx: BaseModel | None) -> int:
        captured.append((inp, ctx))
        return inp + 1

    agent = StubAgent([0])
    branches = {"a": Pipeline.from_step(Step.model_validate({"name": "a", "agent": agent}))}
    step = Step.branch_on(
        name="cond_in_map",
        condition_callable=lambda out, ctx: "a",
        branches=branches,
        branch_input_mapper=inp_map,
    )
    runner = create_test_flujo(step, context_model=FlagCtx)
    await gather_result(runner, 1)
    assert captured and captured[0][0] == 1
    assert isinstance(captured[0][1], BaseModel)
    assert agent.inputs[0] == 2


@pytest.mark.asyncio
async def test_conditional_step_branch_output_mapper_flow() -> None:
    captured: list[tuple[int, str, BaseModel | None]] = []

    def out_map(out: int, key: str, ctx: BaseModel | None) -> int:
        captured.append((out, key, ctx))
        return out * 10

    agent = StubAgent([1])
    branches = {"a": Pipeline.from_step(Step.model_validate({"name": "a", "agent": agent}))}
    step = Step.branch_on(
        name="cond_out_map",
        condition_callable=lambda out, ctx: "a",
        branches=branches,
        branch_output_mapper=out_map,
    )
    runner = create_test_flujo(step, context_model=FlagCtx)
    result = await gather_result(runner, 0)
    step_result = result.step_history[-1]
    assert step_result.output == 10
    assert captured and captured[0][0] == 1 and captured[0][1] == "a"
    assert isinstance(captured[0][2], BaseModel)


@pytest.mark.asyncio
async def test_conditional_step_mappers_with_context_modification() -> None:
    class Ctx2(BaseModel):
        val: int = 0

    def inp_map(inp: int, ctx: Ctx2 | None) -> int:
        if ctx:
            ctx.val += 1
        return inp

    def out_map(out: int, key: str, ctx: Ctx2 | None) -> int:
        if ctx:
            ctx.val += 1
        return out

    agent = StubAgent([0])
    branches = {"a": Pipeline.from_step(Step.model_validate({"name": "a", "agent": agent}))}
    step = Step.branch_on(
        name="cond_ctx_mod",
        condition_callable=lambda out, ctx: "a",
        branches=branches,
        branch_input_mapper=inp_map,
        branch_output_mapper=out_map,
    )
    runner = create_test_flujo(step, context_model=Ctx2)
    result = await gather_result(runner, 0)
    assert result.final_pipeline_context.val == 2


@pytest.mark.asyncio
async def test_conditional_step_default_mapper_behavior() -> None:
    agent = StubAgent(["ok"])
    branches = {"a": Pipeline.from_step(Step.model_validate({"name": "a", "agent": agent}))}
    step = Step.branch_on(
        name="cond_default_map",
        condition_callable=lambda out, ctx: "a",
        branches=branches,
    )
    runner = create_test_flujo(step)
    result = await gather_result(runner, "hi")
    step_result = result.step_history[-1]
    assert agent.inputs[0] == "hi"
    assert step_result.output == "ok"


@pytest.mark.asyncio
async def test_conditional_step_overall_span(monkeypatch) -> None:
    spans: list[str] = []

    class FakeSpan:
        def __init__(self, name: str) -> None:
            spans.append(name)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    from unittest.mock import Mock
    from flujo.infra import telemetry

    mock_logfire = Mock(span=lambda name: FakeSpan(name))
    monkeypatch.setattr(telemetry, "logfire", mock_logfire)

    branches = {
        "a": Pipeline.from_step(Step.model_validate({"name": "a", "agent": StubAgent(["A"])}))
    }
    branch_step = Step.branch_on(
        name="span_cond",
        condition_callable=lambda out, ctx: "a",
        branches=branches,
    )
    runner = create_test_flujo(branch_step)
    await gather_result(runner, "in")
    assert "span_cond" in spans


@pytest.mark.asyncio
async def test_conditional_step_branch_selection_logging_and_span_attributes(monkeypatch) -> None:
    infos: list[str] = []
    spans: list[str] = []
    attrs: list[tuple[str, Any]] = []

    class FakeSpan:
        def __init__(self, name: str) -> None:
            spans.append(name)
            self._attrs: dict[str, Any] = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def set_attribute(self, key: str, value: Any) -> None:
            attrs.append((key, value))
            self._attrs[key] = value

    from unittest.mock import Mock
    from flujo.infra import telemetry

    mock_logfire = Mock(
        span=lambda name: FakeSpan(name),
        info=lambda msg, *a, **k: infos.append(msg),
        warn=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    monkeypatch.setattr(telemetry, "logfire", mock_logfire)

    branches = {
        "a": Pipeline.from_step(Step.model_validate({"name": "step_a", "agent": StubAgent(["A"])}))
    }
    branch_step = Step.branch_on(
        name="cond_span",
        condition_callable=lambda out, ctx: "a",
        branches=branches,
    )
    runner = create_test_flujo(branch_step)
    await gather_result(runner, "in")
    assert any("Condition evaluated to branch key 'a'" in m for m in infos)
    assert any("Executing branch for key 'a'" in m for m in infos)
    assert ("executed_branch_key", "a") in attrs
    assert any("step_a" in s for s in spans)


@pytest.mark.asyncio
async def test_conditional_step_no_branch_match_logging(monkeypatch) -> None:
    warns: list[str] = []

    class FakeSpan:
        def __init__(self, name: str) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def set_attribute(self, key: str, value: Any) -> None:
            pass

    from unittest.mock import Mock
    from flujo.infra import telemetry

    mock_logfire = Mock(
        span=lambda name: FakeSpan(name),
        info=lambda *a, **k: None,
        warn=lambda msg, *a, **k: warns.append(msg),
        error=lambda *a, **k: None,
    )
    monkeypatch.setattr(telemetry, "logfire", mock_logfire)

    branches = {
        "a": Pipeline.from_step(Step.model_validate({"name": "a", "agent": StubAgent(["A"])}))
    }
    step = Step.branch_on(
        name="no_branch",
        condition_callable=lambda out, ctx: "b",
        branches=branches,
    )
    result = await gather_result(create_test_flujo(step), "in")
    assert result.step_history[-1].success is False
    assert any("No branch" in w for w in warns)


@pytest.mark.asyncio
async def test_conditional_step_error_logging_in_callables(monkeypatch) -> None:
    errors: list[str] = []

    class FakeSpan:
        def __init__(self, name: str) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def set_attribute(self, key: str, value: Any) -> None:
            pass

    from unittest.mock import Mock
    from flujo.infra import telemetry

    mock_logfire = Mock(
        span=lambda name: FakeSpan(name),
        info=lambda *a, **k: None,
        warn=lambda *a, **k: None,
        error=lambda msg, *a, **k: errors.append(msg),
    )
    monkeypatch.setattr(telemetry, "logfire", mock_logfire)

    def bad_condition(_: str, __: BaseModel | None) -> str:
        raise RuntimeError("cond boom")

    step = Step.branch_on(
        name="cond_err",
        condition_callable=bad_condition,
        branches={
            "a": Pipeline.from_step(Step.model_validate({"name": "a", "agent": StubAgent(["A"])}))
        },
    )
    await gather_result(create_test_flujo(step), "in")
    assert any("cond boom" in m for m in errors)


@pytest.mark.asyncio
async def test_hitl_in_conditional_branch_pauses_and_resumes() -> None:
    """Test that a human-in-the-loop step in a conditional branch correctly pauses and resumes."""
    classify = Step.model_validate({"name": "classify", "agent": StubAgent(["clarify"])})
    branches = {
        "clarify": Pipeline.from_step(
            Step.human_in_the_loop("clarification", message_for_user="Please clarify.")
        ),
        "ok": Pipeline.from_step(
            Step.model_validate({"name": "ok", "agent": StubAgent(["All good"])})
        ),
    }
    branch_step = Step.branch_on(
        name="branch",
        condition_callable=lambda out, ctx: out,
        branches=branches,
    )
    runner = create_test_flujo(classify >> branch_step)
    # First run should pause for human input
    paused = await gather_result(runner, "input")
    ctx = paused.final_pipeline_context
    assert ctx.scratchpad["status"] == "paused"
    assert ctx.scratchpad["pause_message"] == "Please clarify."
    # Resume with human input
    resumed = await runner.resume_async(paused, "clarified!")
    assert resumed.step_history[-1].output == "clarified!"
    assert ctx.scratchpad["status"] == "completed"  # Should be completed after resume
    assert len(resumed.final_pipeline_context.hitl_history) == 1
    record = resumed.final_pipeline_context.hitl_history[0]
    assert record.message_to_human == "Please clarify."
    assert record.human_response == "clarified!"
