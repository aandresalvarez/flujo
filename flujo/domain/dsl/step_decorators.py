"""Step decorator factories - extracted from step.py to reduce file size.

This module provides the @step and @adapter_step decorators for creating
Step instances from async callables.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Coroutine,
    Optional,
    TYPE_CHECKING,
    overload,
)

try:
    from typing import ParamSpec, Concatenate
except ImportError:
    from typing_extensions import ParamSpec, Concatenate  # type: ignore[assignment]

if TYPE_CHECKING:
    from .step import Step, StepConfig, ExecutionMode
    from ..processors import AgentProcessors

# Type variables matching step.py
from typing import TypeVar

StepInT = TypeVar("StepInT")
StepOutT = TypeVar("StepOutT")
P = ParamSpec("P")

__all__ = ["step", "adapter_step"]


@overload
def step(
    func: Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]],
    *,
    name: str | None = None,
    updates_context: bool = False,
    processors: Optional["AgentProcessors"] = None,
    persist_feedback_to_context: Optional[str] = None,
    persist_validation_results_to: Optional[str] = None,
    is_adapter: bool = False,
    adapter_id: str | None = None,
    adapter_allow: str | None = None,
    **config_kwargs: Any,
) -> "Step[StepInT, StepOutT]": ...


@overload
def step(
    *,
    updates_context: bool = False,
    validate_fields: bool = False,
    sink_to: str | None = None,
    name: Optional[str] = None,
    config: Optional["StepConfig"] = None,
    execution_mode: "ExecutionMode | None" = None,
    max_retries: int | None = None,
    timeout_s: float | None = None,
    adapter_id: str | None = None,
    adapter_allow: str | None = None,
    **config_kwargs: Any,
) -> Callable[
    [Callable[Concatenate[Any, P], Coroutine[Any, Any, Any]]],
    "Step[StepInT, StepOutT]",
]: ...


def step(
    func: (Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]] | None) = None,
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
    **config_kwargs: Any,
) -> Any:
    """Decorator / factory for creating :class:`Step` instances from async callables."""
    # Import here to avoid circular imports
    from .step import Step, StepConfig as _StepConfig

    def decorator(
        fn: Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]],
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

        merged_config_kwargs: dict[str, Any] = {}
        if config is not None:
            merged_config_kwargs.update(config.model_dump())
        merged_config_kwargs.update(config_kwargs)
        if execution_mode is not None:
            merged_config_kwargs["execution_mode"] = execution_mode
        if max_retries is not None:
            merged_config_kwargs["max_retries"] = max_retries
        if timeout_s is not None:
            merged_config_kwargs["timeout_s"] = timeout_s

        final_config = _StepConfig(**merged_config_kwargs)

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
    func: Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]],
    *,
    name: str | None = None,
    updates_context: bool = False,
    adapter_id: str,
    adapter_allow: str,
    processors: Optional["AgentProcessors"] = None,
    persist_feedback_to_context: Optional[str] = None,
    persist_validation_results_to: Optional[str] = None,
    **config_kwargs: Any,
) -> "Step[StepInT, StepOutT]": ...


@overload
def adapter_step(
    *,
    name: str | None = None,
    updates_context: bool = False,
    adapter_id: str,
    adapter_allow: str,
    processors: Optional["AgentProcessors"] = None,
    persist_feedback_to_context: Optional[str] = None,
    persist_validation_results_to: Optional[str] = None,
    **config_kwargs: Any,
) -> Callable[
    [Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]]],
    "Step[StepInT, StepOutT]",
]: ...


def adapter_step(
    func: (Callable[Concatenate[StepInT, P], Coroutine[Any, Any, StepOutT]] | None) = None,
    **kwargs: Any,
) -> Any:
    """Alias for :func:`step` that marks the created step as an adapter."""
    adapter_id = kwargs.pop("adapter_id", None)
    adapter_allow = kwargs.pop("adapter_allow", None)

    if adapter_id is None or adapter_allow is None:
        raise ValueError("adapter_step requires adapter_id and adapter_allow (allowlist token).")

    if func is None:
        return step(
            is_adapter=True,
            adapter_id=adapter_id,
            adapter_allow=adapter_allow,
            **kwargs,
        )
    return step(
        func,
        is_adapter=True,
        adapter_id=adapter_id,
        adapter_allow=adapter_allow,
        **kwargs,
    )
