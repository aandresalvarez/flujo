"""
Agent implementations for the DSL.
"""

from typing import Any, Callable, TypeVar
import inspect
from ..models import BaseModel
from flujo.domain.resources import AppResources

StepInT = TypeVar("StepInT")
StepOutT = TypeVar("StepOutT")
P = TypeVar("P", bound=tuple[Any, ...])


class _CallableAgent:
    """Agent that wraps a callable function for execution in StandardStep."""

    def __init__(self, callable_func: Callable[..., Any]):
        self._step_callable = callable_func
        # Analyze signature for injection
        from ...signature_tools import analyze_signature

        self._injection_spec = analyze_signature(callable_func)
        # Store the original function signature for parameter names
        self._original_sig = inspect.signature(callable_func)

    async def run(
        self,
        data: Any,
        *,
        context: BaseModel | None = None,
        resources: AppResources | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the wrapped callable with proper argument injection."""
        # Build the arguments to pass to the callable
        call_args: list[Any] = []
        callable_kwargs: dict[str, Any] = {}

        # Handle the first parameter (data)
        first_param = next(iter(self._original_sig.parameters.values()))
        if first_param.kind is inspect.Parameter.POSITIONAL_ONLY:
            call_args.append(data)
        else:
            callable_kwargs[first_param.name] = data

        # Add the injected arguments if the callable needs them
        if self._injection_spec.needs_context and context is not None:
            callable_kwargs["context"] = context
        if self._injection_spec.needs_resources and resources is not None:
            callable_kwargs["resources"] = resources

        # Add any additional kwargs
        callable_kwargs.update(kwargs)

        # Call the original function directly
        return await self._step_callable(*call_args, **callable_kwargs)


__all__ = ["_CallableAgent"]
