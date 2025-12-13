from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import TypeAdapter, ValidationError

from ...domain.commands import AgentCommand, AskHumanCommand, FinishCommand, RunAgentCommand
from ...domain.dsl.step import HumanInTheLoopStep, Step
from ...domain.models import (
    ConversationRole,
    ConversationTurn,
    ExecutedCommandLog,
    HumanInteraction,
    PipelineContext,
    PipelineResult,
    StepResult,
)
from ...exceptions import ResumeError
from ...utils.context import set_nested_context_field

_CtxT = TypeVar("_CtxT", bound=PipelineContext)


class ResumeOrchestrator(Generic[_CtxT]):
    """Encapsulates pause/resume bookkeeping for HITL workflows."""

    def __init__(
        self,
        pipeline: Any,
        *,
        trace_manager: Any | None = None,
        agent_command_adapter: TypeAdapter[AgentCommand] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._trace_manager = trace_manager
        self._agent_command_adapter = agent_command_adapter

    def validate_resume(self, paused_result: PipelineResult[_CtxT]) -> _CtxT:
        ctx = paused_result.final_pipeline_context
        if ctx is None:
            raise ResumeError("Cannot resume pipeline without context")
        if getattr(ctx, "status", None) != "paused":
            raise ResumeError("Pipeline is not paused")
        return ctx

    def resolve_paused_step(
        self,
        paused_result: PipelineResult[_CtxT],
        ctx: PipelineContext,
        human_input: Any,
    ) -> tuple[int, Step[Any, Any]]:
        pipeline = self._pipeline
        if pipeline is None or not getattr(pipeline, "steps", None):
            raise ResumeError("No steps remaining to resume")

        steps = list(getattr(pipeline, "steps", []) or [])
        start_idx = len(paused_result.step_history)
        hitl_index_resolved = False

        # Prefer explicit HITL pause metadata when available.
        # This avoids brittle heuristics based on step_history length, which may include
        # placeholder StepResults for paused HITL steps.
        try:
            hd = getattr(ctx, "hitl_data", None)
            last_hitl_step = hd.get("last_hitl_step") if isinstance(hd, dict) else None
            if isinstance(last_hitl_step, str) and last_hitl_step:
                for idx, st in enumerate(steps):
                    if (
                        isinstance(st, HumanInTheLoopStep)
                        and getattr(st, "name", None) == last_hitl_step
                    ):
                        start_idx = idx
                        hitl_index_resolved = True
                        break
        except Exception:
            hitl_index_resolved = False

        # Heuristic: if step_outputs tracks executed top-level steps, resume after the last one.
        # Skip this when we have explicit HITL pause metadata.
        if not hitl_index_resolved:
            try:
                executed_steps = getattr(ctx, "step_outputs", {})
                if isinstance(executed_steps, dict) and executed_steps:
                    last_executed = list(executed_steps.keys())[-1]
                    for idx, st in enumerate(steps):
                        if getattr(st, "name", None) == last_executed:
                            start_idx = min(idx + 1, len(steps) - 1)
                            break
            except Exception:
                pass

        if start_idx >= len(steps):
            # Common pause shape: the paused step is the last (or only) top-level step
            # (e.g., LoopStep pausing internally). In that case, resume from the last step.
            if getattr(ctx, "status", None) == "paused" and steps:
                start_idx = len(steps) - 1
            else:
                raise ResumeError("No steps remaining to resume")

        paused_step = steps[start_idx]

        if getattr(ctx, "status", None) == "paused":
            # Always attach the provided human input to the typed user_input field for downstream steps.
            if hasattr(ctx, "user_input"):
                ctx.user_input = human_input
            # Only set paused_step_input for HITL steps when it's not already holding a pending command.
            if (
                isinstance(paused_step, HumanInTheLoopStep)
                and getattr(ctx, "paused_step_input", None) is None
            ):
                if hasattr(ctx, "paused_step_input"):
                    ctx.paused_step_input = human_input

        return start_idx, paused_step

    def coerce_human_input(self, paused_step: Step[Any, Any], human_input: Any) -> Any:
        if isinstance(paused_step, HumanInTheLoopStep) and paused_step.input_schema is not None:
            return paused_step.input_schema.model_validate(human_input)
        return human_input

    def record_hitl_interaction(
        self,
        ctx: PipelineContext,
        human_input: Any,
        pause_message: str | None,
    ) -> None:
        ctx.hitl_history.append(
            HumanInteraction(
                message_to_human=pause_message or "",
                human_response=human_input,
            )
        )
        # Use typed field instead of scratchpad for status
        ctx.status = "running"

    def update_conversation_history(
        self,
        ctx: PipelineContext,
        human_input: Any,
        pause_message: str | None,
    ) -> None:
        try:
            hist = getattr(ctx, "conversation_history", None)
            if not isinstance(hist, list):
                setattr(ctx, "conversation_history", [])
                hist = ctx.conversation_history
            if pause_message:
                last_content = hist[-1].content if hist else None
                if last_content != pause_message:
                    hist.append(
                        ConversationTurn(
                            role=ConversationRole.assistant, content=str(pause_message)
                        )
                    )
            text = str(human_input)
            last_content = hist[-1].content if hist else None
            if text and text != last_content:
                hist.append(ConversationTurn(role=ConversationRole.user, content=text))
        except Exception:
            pass

    def apply_sink_to(
        self,
        ctx: PipelineContext,
        paused_step: Step[Any, Any],
        human_input: Any,
    ) -> None:
        sink_to = getattr(paused_step, "sink_to", None)
        if not sink_to:
            return
        try:
            set_nested_context_field(ctx, sink_to, human_input)
        except Exception:
            pass

    def update_steps_map(
        self,
        ctx: PipelineContext,
        paused_step: Step[Any, Any],
        human_input: Any,
    ) -> None:
        try:
            # Write to step_outputs (primary)
            if hasattr(ctx, "step_outputs"):
                current_outputs = getattr(ctx, "step_outputs", {})
                if not isinstance(current_outputs, dict):
                    current_outputs = {}
                    ctx.step_outputs = current_outputs
                steps_map = current_outputs
            else:
                return
            val = human_input
            if isinstance(val, bytes):
                try:
                    val = val.decode("utf-8", errors="ignore")
                except Exception:
                    val = str(val)
            else:
                val = str(val)
            if len(val) > 1024:
                val = val[:1024]
            steps_map[getattr(paused_step, "name", "")] = val
        except Exception:
            pass

    def record_pending_command_log(
        self,
        ctx: PipelineContext,
        paused_step: Step[Any, Any],
        human_input: Any,
    ) -> None:
        pending = getattr(ctx, "paused_step_input", None)
        if pending is None:
            return
        # Consume pending command payload so subsequent resumes don't re-log it.
        try:
            if hasattr(ctx, "paused_step_input"):
                ctx.paused_step_input = None
        except Exception:
            pass
        adapter = self._agent_command_adapter
        pending_cmd: AgentCommand | AskHumanCommand | FinishCommand | RunAgentCommand | None
        pending_cmd = None
        try:
            if isinstance(pending, (RunAgentCommand, AskHumanCommand, FinishCommand)):
                pending_cmd = pending
            elif adapter is not None:
                pending_cmd = adapter.validate_python(pending)
        except ValidationError:
            pending_cmd = None
        except Exception:
            pending_cmd = None

        try:
            if pending_cmd is not None:
                log_entry = ExecutedCommandLog(
                    turn=len(ctx.command_log) + 1,
                    generated_command=pending_cmd,
                    execution_result=human_input,
                )
            else:
                log_entry = ExecutedCommandLog(
                    turn=len(ctx.command_log) + 1,
                    generated_command=AskHumanCommand(
                        question=getattr(ctx, "pause_message", None) or "Paused"
                    ),
                    execution_result=human_input,
                )
            ctx.command_log.append(log_entry)
            if hasattr(ctx, "loop_last_output"):
                ctx.loop_last_output = log_entry
        except Exception:
            pass

    def build_step_result(self, paused_step: Step[Any, Any], human_input: Any) -> StepResult:
        return StepResult(
            name=paused_step.name,
            output=human_input,
            success=True,
            attempts=1,
        )

    def add_trace_event(self, human_input: Any) -> None:
        tm = self._trace_manager
        if tm is None:
            return
        try:
            summary = str(human_input)
            if isinstance(summary, str) and len(summary) > 500:
                summary = summary[:500] + "..."
            tm.add_event("flujo.resumed", {"human_input": summary})
        except Exception:
            pass
