from __future__ import annotations
from .step_policies import StateMachinePolicyExecutor

"""Backward-compatible shim for StateMachine policy executor.

The implementation now lives in `flujo.application.core.step_policies`.
This module re-exports `StateMachinePolicyExecutor` to preserve imports
in existing tests and external code.
"""

__all__ = ["StateMachinePolicyExecutor"]
