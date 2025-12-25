import inspect

import pytest

from flujo.application.core.step_policies import (
    DefaultAgentStepExecutor,
    DefaultCacheStepExecutor,
    DefaultConditionalStepExecutor,
    DefaultDynamicRouterStepExecutor,
    DefaultHitlStepExecutor,
    DefaultImportStepExecutor,
    DefaultLoopStepExecutor,
    DefaultParallelStepExecutor,
    DefaultTreeSearchStepExecutor,
    DefaultSimpleStepExecutor,
)


@pytest.mark.parametrize(
    "policy_cls",
    [
        DefaultSimpleStepExecutor,
        DefaultAgentStepExecutor,
        DefaultCacheStepExecutor,
        DefaultConditionalStepExecutor,
        DefaultParallelStepExecutor,
        DefaultLoopStepExecutor,
        DefaultDynamicRouterStepExecutor,
        DefaultHitlStepExecutor,
        DefaultImportStepExecutor,
        DefaultTreeSearchStepExecutor,
    ],
)
def test_policy_execute_signature_uses_frame(policy_cls: type) -> None:
    """All Default* executors must be frame-first to avoid legacy regression."""
    sig = inspect.signature(policy_cls.execute)
    params = list(sig.parameters.keys())
    assert params[:3] == ["self", "core", "frame"], f"{policy_cls.__name__} signature drifted"
