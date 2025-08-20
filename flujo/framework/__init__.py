from __future__ import annotations

# Framework initialization: register first-class primitives and their policies

from .registry import register_step_type, register_policy


def _register_core_primitives() -> None:
    # Import here to avoid import cycles during package initialization
    from ..domain.dsl.state_machine import StateMachineStep
    from ..application.core.step_policies import StateMachinePolicyExecutor

    register_step_type(StateMachineStep)
    register_policy(StateMachineStep, StateMachinePolicyExecutor())


# Perform registration on import
_register_core_primitives()
