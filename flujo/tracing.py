from __future__ import annotations

from typing import Any, Literal

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
            Console(highlight=False)
            if colorized
            else Console(no_color=True, highlight=False)
        )

    async def hook(self, **kwargs: Any) -> None:
        event = kwargs.get("event_name")
        if event == "pre_step":
            step = kwargs.get("step")
            step_input = kwargs.get("step_input")
            title = f"Step Start: {step.name if step else ''}"
            if self.level == "debug" and self.log_inputs:
                body = Text(repr(step_input))
            else:
                body = Text("running")
            self.console.print(Panel(body, title=title))
        elif event == "post_step":
            step_result = kwargs.get("step_result")
            if step_result is None:
                return
            title = f"Step End: {step_result.name}"
            status = "SUCCESS" if step_result.success else "FAILED"
            color = "green" if step_result.success else "red"
            body_text = Text(f"Status: {status}", style=f"bold {color}")
            if self.level == "debug" and self.log_outputs:
                body_text.append(f"\nOutput: {repr(step_result.output)}")
            self.console.print(Panel(body_text, title=title))
        elif event == "pre_run":
            initial_input = kwargs.get("initial_input")
            title = "Pipeline Start"
            details = Text(f"Input: {initial_input!r}")
            self.console.print(Panel(details, title=title, border_style="bold blue"))
        elif event == "post_run":
            pipeline_result = kwargs.get("pipeline_result")
            if pipeline_result is None:
                return
            title = "Pipeline End"
            is_success = all(s.success for s in pipeline_result.step_history)
            status_text = "✅ COMPLETED" if is_success else "❌ FAILED"
            status_style = "bold green" if is_success else "bold red"
            details = Text()
            details.append(f"Final Status: {status_text}\n", style=status_style)
            details.append(
                f"Total Steps Executed: {len(pipeline_result.step_history)}\n"
            )
            details.append(f"Total Cost: ${pipeline_result.total_cost_usd:.6f}")
            self.console.print(Panel(details, title=title, border_style="bold blue"))
        elif event == "on_step_failure":
            step_result = kwargs.get("step_result")
            if step_result is None:
                return
            title = f"Step End: {step_result.name}"
            details = Text(
                f"Status: FAILED\nFeedback: {step_result.feedback}",
                style="red",
            )
            self.console.print(Panel(details, title=title, border_style="bold red"))
        else:
            self.console.print(Panel(Text(str(event)), title="Unknown tracer event"))
