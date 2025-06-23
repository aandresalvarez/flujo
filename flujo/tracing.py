from __future__ import annotations

from typing import Any, Callable, Dict, Literal, cast

from .domain.events import (
    HookPayload,
    PreRunPayload,
    PostRunPayload,
    PreStepPayload,
    PostStepPayload,
    OnStepFailurePayload,
)

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

__all__ = ["ConsoleTracer"]


class ConsoleTracer:
    """Configurable tracer that prints rich output to the console."""

    def __init__(
        self,
        *,
        level: Literal["info", "debug"] = "debug",
        log_inputs: bool = True,
        log_outputs: bool = True,
        colorized: bool = True,
    ) -> None:
        self.level = level
        self.log_inputs = log_inputs
        self.log_outputs = log_outputs
        self.console = (
            Console(highlight=False) if colorized else Console(no_color=True, highlight=False)
        )
        self.event_handlers: Dict[str, Callable[[HookPayload], Any]] = {
            "pre_run": cast(Callable[[HookPayload], Any], self._handle_pre_run),
            "post_run": cast(Callable[[HookPayload], Any], self._handle_post_run),
            "pre_step": cast(Callable[[HookPayload], Any], self._handle_pre_step),
            "post_step": cast(Callable[[HookPayload], Any], self._handle_post_step),
            "on_step_failure": cast(Callable[[HookPayload], Any], self._handle_on_step_failure),
        }

    def _handle_pre_run(self, payload: PreRunPayload) -> None:
        initial_input = payload.initial_input
        title = "Pipeline Start"
        details = Text(f"Input: {initial_input!r}")
        self.console.print(Panel(details, title=title, border_style="bold blue"))

    def _handle_post_run(self, payload: PostRunPayload) -> None:
        pipeline_result = payload.pipeline_result
        title = "Pipeline End"
        is_success = all(s.success for s in pipeline_result.step_history)
        status_text = "✅ COMPLETED" if is_success else "❌ FAILED"
        status_style = "bold green" if is_success else "bold red"
        details = Text()
        details.append(f"Final Status: {status_text}\n", style=status_style)
        details.append(f"Total Steps Executed: {len(pipeline_result.step_history)}\n")
        details.append(f"Total Cost: ${pipeline_result.total_cost_usd:.6f}")
        self.console.print(Panel(details, title=title, border_style="bold blue"))

    def _handle_pre_step(self, payload: PreStepPayload) -> None:
        step = payload.step
        step_input = payload.step_input
        title = f"Step Start: {step.name if step else ''}"
        if self.level == "debug" and self.log_inputs:
            body = Text(repr(step_input))
        else:
            body = Text("running")
        self.console.print(Panel(body, title=title))

    def _handle_post_step(self, payload: PostStepPayload) -> None:
        step_result = payload.step_result
        title = f"Step End: {step_result.name}"
        status = "SUCCESS" if step_result.success else "FAILED"
        color = "green" if step_result.success else "red"
        body_text = Text(f"Status: {status}", style=f"bold {color}")
        if self.level == "debug" and self.log_outputs:
            body_text.append(f"\nOutput: {repr(step_result.output)}")
        self.console.print(Panel(body_text, title=title))

    def _handle_on_step_failure(self, payload: OnStepFailurePayload) -> None:
        step_result = payload.step_result
        title = f"Step Failure: {step_result.name}"
        details = Text(
            f"Status: FAILED\nFeedback: {step_result.feedback}",
            style="red",
        )
        self.console.print(Panel(details, title=title, border_style="bold red"))

    async def hook(self, payload: HookPayload) -> None:
        handler = self.event_handlers.get(payload.event_name)
        if handler:
            import inspect

            if inspect.iscoroutinefunction(handler):
                await handler(payload)
            else:
                handler(payload)
        else:
            self.console.print(Panel(Text(str(payload.event_name)), title="Unknown tracer event"))
