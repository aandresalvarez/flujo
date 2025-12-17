"""Step decorator factories - extracted from step.py to reduce file size.

This module provides the @step and @adapter_step decorators for creating
Step instances from async callables.
"""

from __future__ import annotations

from typing import Callable, Concatenate, Coroutine, Optional, TYPE_CHECKING, ParamSpec, overload

if TYPE_CHECKING:
    from .step import ExecutionMode
    from ..processors import AgentProcessors

# Import at module scope to avoid per-call runtime imports; safe because step.py
# imports this module only after defining Step/StepConfig.
from .step import Step, StepConfig

# Type variables matching step.py
from typing import TypeVar

StepInT = TypeVar("StepInT")
StepOutT = TypeVar("StepOutT")
P = ParamSpec("P")

__all__ = ["step", "adapter_step"]


@overload
def step(
    func: Callable[Concatenate[StepInT, P], Coroutine[object, object, StepOutT]],
    *,
    name: str | None = None,
    updates_context: bool = False,
    validate_fields: bool = False,
    sink_to: str | None = None,
    config: Optional["StepConfig"] = None,
    execution_mode: "ExecutionMode | None" = None,
    max_retries: int | None = None,
    timeout_s: float | None = None,
    processors: Optional["AgentProcessors"] = None,
    persist_feedback_to_context: Optional[str] = None,
    persist_validation_results_to: Optional[str] = None,
    is_adapter: bool = False,
    adapter_id: str | None = None,
    adapter_allow: str | None = None,
    **config_kwargs: object,
) -> "Step[StepInT, StepOutT]": ...


@overload
def step(
    func: None = None,
    *,
    updates_context: bool = False,
    validate_fields: bool = False,
    sink_to: str | None = None,
    name: Optional[str] = None,
    config: Optional["StepConfig"] = None,
    execution_mode: "ExecutionMode | None" = None,
    max_retries: int | None = None,
    timeout_s: float | None = None,
    processors: Optional["AgentProcessors"] = None,
    persist_feedback_to_context: Optional[str] = None,
    persist_validation_results_to: Optional[str] = None,
    is_adapter: bool = False,
    adapter_id: str | None = None,
    adapter_allow: str | None = None,
    **config_kwargs: object,
) -> Callable[
    [Callable[Concatenate[StepInT, P], Coroutine[object, object, StepOutT]]],
    "Step[StepInT, StepOutT]",
]: ...


def step(
    func: Callable[..., Coroutine[object, object, object]] | None = None,
    *,
    name: str | None = None,
    updates_context: bool = False,
    validate_fields: bool = False,
    sink_to: str | None = None,
    config: "StepConfig | None" = None,
    execution_mode: "ExecutionMode | None" = None,
    max_retries: int | None = None,
    timeout_s: float | None = None,
    processors: Optional["AgentProcessors"] = None,
    persist_feedback_to_context: Optional[str] = None,
    persist_validation_results_to: Optional[str] = None,
    is_adapter: bool = False,
    adapter_id: str | None = None,
    adapter_allow: str | None = None,
    **config_kwargs: object,
) -> object:
    """Decorator / factory for creating :class:`Step` instances from async callables."""

    def decorator(
        fn: Callable[Concatenate[StepInT, P], Coroutine[object, object, StepOutT]],
    ) -> "Step[StepInT, StepOutT]":
        if config is not None and (
            execution_mode is not None
            or max_retries is not None
            or timeout_s is not None
            or config_kwargs
        ):
            import warnings as _warnings

            _warnings.warn(
                "Both `config` and additional config parameters were provided to @step; "
                "explicit parameters will override the StepConfig values.",
                UserWarning,
                stacklevel=2,
            )

        merged_config_kwargs: dict[str, object] = {}
        if config is not None:
            merged_config_kwargs.update(config.model_dump())
        merged_config_kwargs.update(config_kwargs)
        if execution_mode is not None:
            merged_config_kwargs["execution_mode"] = execution_mode
        if max_retries is not None:
            merged_config_kwargs["max_retries"] = max_retries
        if timeout_s is not None:
            merged_config_kwargs["timeout_s"] = timeout_s

        final_config = StepConfig.model_validate(merged_config_kwargs)

        return Step.from_callable(
            fn,
            name=name or fn.__name__,
            updates_context=updates_context,
            validate_fields=validate_fields,
            sink_to=sink_to,
            processors=processors,
            persist_feedback_to_context=persist_feedback_to_context,
            persist_validation_results_to=persist_validation_results_to,
            is_adapter=is_adapter,
            adapter_id=adapter_id,
            adapter_allow=adapter_allow,
            config=final_config,
        )

    # If used without parentheses, func is the callable
    if func is not None:
        return decorator(func)

    return decorator


@overload
def adapter_step(
    func: Callable[Concatenate[StepInT, P], Coroutine[object, object, StepOutT]],
    *,
    name: str | None = None,
    updates_context: bool = False,
    adapter_id: str,
    adapter_allow: str,
    processors: Optional["AgentProcessors"] = None,
    persist_feedback_to_context: Optional[str] = None,
    persist_validation_results_to: Optional[str] = None,
    **config_kwargs: object,
) -> "Step[StepInT, StepOutT]": ...


@overload
def adapter_step(
    func: None = None,
    *,
    name: str | None = None,
    updates_context: bool = False,
    adapter_id: str,
    adapter_allow: str,
    processors: Optional["AgentProcessors"] = None,
    persist_feedback_to_context: Optional[str] = None,
    persist_validation_results_to: Optional[str] = None,
    **config_kwargs: object,
) -> Callable[
    [Callable[Concatenate[StepInT, P], Coroutine[object, object, StepOutT]]],
    "Step[StepInT, StepOutT]",
]: ...


def adapter_step(
    func: Callable[..., Coroutine[object, object, object]] | None = None,
    *,
    name: str | None = None,
    updates_context: bool = False,
    adapter_id: str,
    adapter_allow: str,
    processors: Optional["AgentProcessors"] = None,
    persist_feedback_to_context: Optional[str] = None,
    persist_validation_results_to: Optional[str] = None,
    **config_kwargs: object,
) -> object:
    """Alias for :func:`step` that marks the created step as an adapter."""
    if func is None:
        return step(  # type: ignore[call-overload]
            name=name,
            updates_context=updates_context,
            processors=processors,
            persist_feedback_to_context=persist_feedback_to_context,
            persist_validation_results_to=persist_validation_results_to,
            is_adapter=True,
            adapter_id=adapter_id,
            adapter_allow=adapter_allow,
            **config_kwargs,
        )
    return step(  # type: ignore[call-overload]
        func,
        name=name,
        updates_context=updates_context,
        processors=processors,
        persist_feedback_to_context=persist_feedback_to_context,
        persist_validation_results_to=persist_validation_results_to,
        is_adapter=True,
        adapter_id=adapter_id,
        adapter_allow=adapter_allow,
        **config_kwargs,
    )
