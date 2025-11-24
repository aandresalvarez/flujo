from __future__ import annotations

import asyncio


def test_custom_step_registration_and_yaml_loader() -> None:
    from typing import Any
    from flujo.framework import registry
    from flujo.domain.dsl.step import Step
    from flujo.domain.dsl import Pipeline

    from typing import ClassVar

    class MyStep(Step[Any, Any]):
        kind: ClassVar[str] = "CustomX"

    class MyPolicy:
        async def execute(self, core, frame):  # type: ignore[no-untyped-def]
            from flujo.domain.models import StepResult, Success

            sr = StepResult(name=getattr(frame.step, "name", "custom"), output=None, success=True)
            return Success(step_result=sr)

    # Register type + policy
    registry.register_step_type(MyStep)
    registry.register_policy(MyStep, MyPolicy())

    yaml_text = 'version: "0.1"\nsteps:\n  - kind: CustomX\n    name: X\n'
    p = Pipeline.from_yaml_text(yaml_text)
    assert p is not None
    assert len(p.steps) == 1
    assert isinstance(p.steps[0], MyStep)

    # Execute to ensure policy resolves
    from flujo.application.core.executor_core import ExecutorCore

    core = ExecutorCore()

    async def _run() -> None:
        res = await core._execute_pipeline_via_policies(p, None, None, None, None, None)
        assert res is not None
        assert len(res.step_history) == 1

    asyncio.run(_run())
