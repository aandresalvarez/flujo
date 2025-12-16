from __future__ import annotations

import pytest


def test_register_step_type_and_policy() -> None:
    from flujo.framework import registry
    from flujo.domain.dsl.state_machine import StateMachineStep

    prior_policies = registry.get_registered_policies()
    prior_policy = prior_policies.get(StateMachineStep)

    # Sanity: kind is present
    assert getattr(StateMachineStep, "kind", None) == "StateMachine"

    # Duplicate registration should raise
    with pytest.raises(Exception):
        registry.register_step_type(StateMachineStep)

    # Registering non-Step should raise
    class _NotAStep:  # type: ignore[too-few-public-methods]
        kind = "Bad"

    with pytest.raises(Exception):
        registry.register_step_type(_NotAStep)  # type: ignore[arg-type]

    # Policy API accepts any instance; verify it stores
    class _DummyPolicy:  # type: ignore[too-few-public-methods]
        async def execute(self, core, frame):  # pragma: no cover - trivial
            from flujo.domain.models import StepResult, Success

            sr = StepResult(name=getattr(frame.step, "name", "dummy"), output=None, success=True)
            return Success(step_result=sr)

    try:
        registry.register_policy(StateMachineStep, _DummyPolicy())
        policies = registry.get_registered_policies()
        assert StateMachineStep in policies
    finally:
        if prior_policy is not None:
            registry.register_policy(StateMachineStep, prior_policy)
        else:
            registry.unregister_policy(StateMachineStep)
