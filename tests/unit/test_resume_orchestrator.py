from __future__ import annotations

from typing import Any

from pydantic import BaseModel, TypeAdapter

from flujo.application.runner_components import ResumeOrchestrator
from flujo.domain.commands import AgentCommand, AskHumanCommand
from flujo.domain.dsl.step import HumanInTheLoopStep
from flujo.domain.models import PipelineContext, PipelineResult
from flujo.exceptions import ResumeError


class _Schema(BaseModel):
    value: int


class _Ctx(PipelineContext):
    answer: _Schema | None = None


class _Pipeline:
    def __init__(self, steps: list[Any]) -> None:
        self.steps = steps


def test_validate_and_resolve_paused_step_sets_status_and_indices() -> None:
    ctx = _Ctx()
    ctx.status = "paused"
    paused_result: PipelineResult[_Ctx] = PipelineResult(
        final_pipeline_context=ctx, step_history=[]
    )
    step = HumanInTheLoopStep(name="hitl", agent=None)
    orch = ResumeOrchestrator(
        _Pipeline([step]),
        trace_manager=None,
        agent_command_adapter=TypeAdapter(AgentCommand),
    )

    resolved_ctx = orch.validate_resume(paused_result)
    start_idx, resolved_step = orch.resolve_paused_step(paused_result, resolved_ctx, "hi")

    assert resolved_ctx is ctx
    assert start_idx == 0
    assert resolved_step is step
    assert ctx.user_input == "hi"
    assert ctx.paused_step_input == "hi"


def test_coerce_and_context_updates_apply_sink_and_logs() -> None:
    ctx = _Ctx()
    ctx.status = "paused"
    ctx.pause_message = "need input"
    ctx.paused_step_input = AskHumanCommand(question="Q?")
    paused_result: PipelineResult[_Ctx] = PipelineResult(
        final_pipeline_context=ctx, step_history=[]
    )
    step = HumanInTheLoopStep(name="hitl", agent=None, input_schema=_Schema, sink_to="answer")
    orch = ResumeOrchestrator(
        _Pipeline([step]),
        trace_manager=None,
        agent_command_adapter=TypeAdapter(AgentCommand),
    )

    resolved_ctx = orch.validate_resume(paused_result)
    start_idx, paused_step = orch.resolve_paused_step(paused_result, resolved_ctx, {"value": 3})
    human_input = orch.coerce_human_input(paused_step, {"value": 3})
    pause_msg = resolved_ctx.pause_message
    orch.record_hitl_interaction(resolved_ctx, human_input, pause_msg)
    orch.update_conversation_history(resolved_ctx, human_input, pause_msg)
    orch.apply_sink_to(resolved_ctx, paused_step, human_input)
    orch.update_steps_map(resolved_ctx, paused_step, human_input)
    orch.record_pending_command_log(resolved_ctx, paused_step, human_input)

    assert start_idx == 0
    assert resolved_ctx.status == "running"
    assert resolved_ctx.answer == human_input
    assert "hitl" in (resolved_ctx.step_outputs or {})
    assert resolved_ctx.command_log, "Command log should be populated when paused_step_input exists"
    assert resolved_ctx.conversation_history, "Conversation history captures pause/user content"


def test_validate_resume_raises_when_not_paused() -> None:
    ctx = PipelineContext()
    ctx.status = "running"
    paused_result: PipelineResult[PipelineContext] = PipelineResult(
        final_pipeline_context=ctx, step_history=[]
    )
    orch = ResumeOrchestrator(
        _Pipeline([HumanInTheLoopStep(name="hitl", agent=None)]),
        trace_manager=None,
        agent_command_adapter=TypeAdapter(AgentCommand),
    )

    try:
        orch.validate_resume(paused_result)
    except ResumeError:
        return
    assert False, "Expected ResumeError when status is not paused"
