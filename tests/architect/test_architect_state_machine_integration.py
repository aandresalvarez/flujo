from __future__ import annotations

from typing import Any

import pytest


@pytest.mark.asyncio
@pytest.mark.slow
async def test_architect_state_machine_reaches_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLUJO_ARCHITECT_STATE_MACHINE", "1")

    from flujo.architect.builder import build_architect_pipeline
    from flujo.architect.context import ArchitectContext
    from flujo.cli.helpers import create_flujo_runner

    pipeline = build_architect_pipeline()
    ctx_data: dict[str, Any] = {
        "initial_prompt": "Create a simple plan",
        "user_goal": "Generate a sample pipeline",
    }

    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=ctx_data
    )

    result = None
    iterations = 0
    async for item in runner.run_async("Create a simple plan"):
        result = item
        iterations += 1
        if iterations > 20:
            break

    assert result is not None
    final_ctx = getattr(result, "final_pipeline_context", None)
    assert final_ctx is not None
    yaml_text = getattr(final_ctx, "yaml_text", "")
    assert isinstance(yaml_text, str) and yaml_text
    assert "version" in yaml_text and "steps" in yaml_text
