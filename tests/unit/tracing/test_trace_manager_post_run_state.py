from __future__ import annotations

from flujo.tracing.manager import TraceManager
from flujo.domain.events import PreRunPayload, PostRunPayload
from flujo.domain.models import PipelineResult, PipelineContext


def test_trace_manager_post_run_marks_paused() -> None:
    tm = TraceManager()

    # pre_run to create root span
    pre = PreRunPayload(
        event_name="pre_run",
        initial_input="x",
        context=None,
    )

    import asyncio

    asyncio.run(tm._handle_pre_run(pre))

    # Build paused result
    ctx = PipelineContext()
    ctx.status = "paused"
    res: PipelineResult[PipelineContext] = PipelineResult()
    res.final_pipeline_context = ctx
    res.success = False

    post = PostRunPayload(event_name="post_run", pipeline_result=res, context=ctx)
    asyncio.run(tm._handle_post_run(post))

    assert tm._root_span is not None  # noqa: S101
    # Paused pipelines should be reflected as paused rather than failed.
    assert tm._root_span.status == "paused"  # noqa: S101
