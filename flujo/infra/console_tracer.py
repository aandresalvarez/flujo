from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal

from ..domain.events import (
    HookPayload,
    PreRunPayload,
    PostRunPayload,
    PreStepPayload,
    PostStepPayload,
    OnStepFailurePayload,
)

from typing import Any as _Any

__all__ = ["ConsoleTracer"]


@dataclass(frozen=True, slots=True)
class _PlainPanel:
    title: str
    body: str

    def __str__(self) -> str:
        if self.body:
            return f"{self.title} :: {self.body}"
        return self.title


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
        """Create the tracer.

        Parameters
        ----------
        level:
            Output verbosity; either ``"info"`` or ``"debug"``.
        log_inputs:
            Whether to print step inputs when ``level`` is ``"debug"``.
        log_outputs:
            Whether to print step outputs when ``level`` is ``"debug"``.
        colorized:
            If ``True`` use colored output via Rich.
        """

        self.level = level
        self.log_inputs = log_inputs
        self.log_outputs = log_outputs
        # Lazy-import Rich to avoid hard dependency at import time
        try:
            from rich.console import Console as _RConsole

            self.console: _Any = (
                _RConsole(highlight=False)
                if colorized
                else _RConsole(no_color=True, highlight=False)
            )
            # Keep Panel/Text types for later use
            from rich.panel import Panel as _RPanel
            from rich.text import Text as _RText

            self._RPanel: _Any = _RPanel
            self._RText: _Any = _RText
            self._rich_available = True
        except ModuleNotFoundError:
            # Minimal stub with .print compatible with Console
            class _Stub:
                def print(self, msg: _Any, *args: _Any, **kwargs: _Any) -> None:  # noqa: D401
                    try:
                        import typer as _ty

                        _ty.echo(str(msg))
                    except Exception:
                        print(str(msg))

            self.console = _Stub()
            self._RPanel = None
            self._RText = None
            self._rich_available = False
        self._depth = 0
        self.event_handlers: Dict[str, Callable[[HookPayload], Any]] = {
            "pre_run": self._handle_pre_run,
            "post_run": self._handle_post_run,
            "pre_step": self._handle_pre_step,
            "post_step": self._handle_post_step,
            "on_step_failure": self._handle_on_step_failure,
        }

    def _emit_panel(self, title: str, body: str) -> None:
        if self._rich_available and self._RPanel is not None and self._RText is not None:
            text = self._RText(body)
            self.console.print(self._RPanel(text, title=title))
        else:
            self.console.print(_PlainPanel(title=title, body=body))

    def _handle_pre_run(self, payload: HookPayload) -> None:
        """Handle the ``pre_run`` event."""
        if not isinstance(payload, PreRunPayload):
            return
        if getattr(payload, "is_background", False):
            return
        initial_input = payload.initial_input
        title = "Pipeline Start"
        self._emit_panel(title, f"Input: {initial_input!r}")
        self._depth = 0

    def _handle_post_run(self, payload: HookPayload) -> None:
        """Handle the ``post_run`` event.

        Correctly reflect paused/failed/completed final state. Previously this handler
        inferred success solely from ``step_history`` which reports ``all([]) == True``
        in paused-early cases, causing a misleading "COMPLETED" panel. We now:
        - Prefer ``pipeline_result.success`` when available (runner sets this).
        - Detect a paused state from ``payload.context.status`` (or
          ``pipeline_result.final_pipeline_context`` as a fallback).
        - Adjust the title and styling accordingly.
        """
        if not isinstance(payload, PostRunPayload):
            return
        if getattr(payload, "is_background", False):
            return
        pipeline_result = payload.pipeline_result

        # Determine final status
        ctx = payload.context or getattr(pipeline_result, "final_pipeline_context", None)
        paused = bool(ctx is not None and getattr(ctx, "status", None) == "paused")

        # Prefer the explicit success flag set by the runner; fall back to step_history
        is_success = bool(getattr(pipeline_result, "success", False))

        if paused:
            title = "Pipeline Paused"
            status_text = "⏸ PAUSED"
            status_style = "bold yellow"
        elif is_success:
            title = "Pipeline End"
            status_text = "✅ COMPLETED"
            status_style = "bold green"
        else:
            title = "Pipeline End"
            status_text = "❌ FAILED"
            status_style = "bold red"

        if self._rich_available and self._RPanel is not None and self._RText is not None:
            details = self._RText()
            details.append(f"Status: {status_text}\n", style=status_style)
            details.append(f"Total Steps Executed: {len(pipeline_result.step_history)}\n")
            details.append(f"Total Cost: ${pipeline_result.total_cost_usd:.6f}")
            self.console.print(self._RPanel(details, title=title, border_style="bold blue"))
        else:
            self._emit_panel(
                title,
                (
                    f"Status: {status_text} :: Steps={len(pipeline_result.step_history)} "
                    f":: Cost=${pipeline_result.total_cost_usd:.6f}"
                ),
            )

    def _handle_pre_step(self, payload: HookPayload) -> None:
        """Handle the ``pre_step`` event."""
        if not isinstance(payload, PreStepPayload):
            return
        if getattr(payload, "is_background", False):
            return
        step = payload.step
        step_input = payload.step_input
        indent = "  " * self._depth
        title = f"{indent}Step Start: {step.name if step else ''}"
        body = repr(step_input) if (self.level == "debug" and self.log_inputs) else "running"
        self._emit_panel(title, body)
        self._depth += 1

    def _handle_post_step(self, payload: HookPayload) -> None:
        """Handle the ``post_step`` event."""
        if not isinstance(payload, PostStepPayload):
            return
        if getattr(payload, "is_background", False):
            return
        step_result = payload.step_result
        self._depth = max(0, self._depth - 1)
        indent = "  " * self._depth
        title = f"{indent}Step End: {step_result.name}"
        status = "SUCCESS" if step_result.success else "FAILED"
        color = "green" if step_result.success else "red"
        if self._rich_available and self._RPanel is not None and self._RText is not None:
            body_text = self._RText(f"Status: {status}", style=f"bold {color}")
            if self.level == "debug" and self.log_outputs:
                body_text.append(f"\nOutput: {repr(step_result.output)}")
            self.console.print(self._RPanel(body_text, title=title))
        else:
            body = f"Status: {status}"
            if self.level == "debug" and self.log_outputs:
                body = f"{body} :: Output: {repr(step_result.output)}"
            self._emit_panel(title, body)

    def _handle_on_step_failure(self, payload: HookPayload) -> None:
        """Handle the ``on_step_failure`` event."""
        if not isinstance(payload, OnStepFailurePayload):
            return
        if getattr(payload, "is_background", False):
            return
        step_result = payload.step_result
        self._depth = max(0, self._depth - 1)
        indent = "  " * self._depth
        title = f"{indent}Step Failure: {step_result.name}"
        if self._rich_available and self._RPanel is not None and self._RText is not None:
            details = self._RText(
                f"Status: FAILED\nFeedback: {step_result.feedback}",
                style="red",
            )
            self.console.print(self._RPanel(details, title=title, border_style="bold red"))
        else:
            self._emit_panel(title, f"Status: FAILED :: Feedback: {step_result.feedback}")

    async def hook(self, payload: HookPayload) -> None:
        """Dispatch hook payloads to the appropriate handler."""
        handler = self.event_handlers.get(payload.event_name)
        if handler:
            import inspect

            if inspect.iscoroutinefunction(handler):
                await handler(payload)
            else:
                handler(payload)
        else:
            if self._rich_available and self._RPanel is not None and self._RText is not None:
                self.console.print(
                    self._RPanel(self._RText(str(payload.event_name)), title="Unknown tracer event")
                )
            else:
                self.console.print(f"Unknown tracer event: {payload.event_name}")
