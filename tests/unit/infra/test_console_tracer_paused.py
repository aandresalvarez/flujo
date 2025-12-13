from __future__ import annotations

from flujo.infra.console_tracer import ConsoleTracer
from flujo.domain.events import PostRunPayload
from flujo.domain.models import PipelineResult, PipelineContext
from rich.console import Console


def test_console_tracer_post_run_paused_displays_paused() -> None:
    tracer = ConsoleTracer()
    # Capture output from tracer
    recorder = Console(record=True)
    tracer.console = recorder

    # Build a paused context and pipeline result
    ctx = PipelineContext()
    ctx.status = "paused"
    ctx.pause_message = "What is your name?"

    result: PipelineResult[PipelineContext] = PipelineResult()
    result.final_pipeline_context = ctx
    result.success = False

    payload = PostRunPayload(event_name="post_run", pipeline_result=result, context=ctx)

    tracer._handle_post_run(payload)

    text = recorder.export_text()
    assert "PAUSED" in text  # noqa: S101
