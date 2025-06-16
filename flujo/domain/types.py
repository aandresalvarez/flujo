"""Shared type aliases for the domain layer."""

from typing import Callable, Coroutine, Any

from .models import PipelineResult, StepResult  # noqa: F401
from .resources import AppResources  # noqa: F401

# A hook is an async callable that receives keyword arguments.
HookCallable = Callable[..., Coroutine[Any, Any, None]]
