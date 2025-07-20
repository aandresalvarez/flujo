"""Custom exceptions for the orchestrator."""

from __future__ import annotations

from typing import Any

from flujo.domain.models import PipelineResult


class OrchestratorError(Exception):
    """Base exception for the application."""

    pass


class SettingsError(OrchestratorError):
    """Raised for configuration-related errors."""

    pass


class OrchestratorRetryError(OrchestratorError):
    """Raised when an agent operation fails after all retries."""

    pass


class RewardModelUnavailable(OrchestratorError):
    """Raised when the reward model is required but unavailable."""

    pass


class FeatureDisabled(OrchestratorError):
    """Raised when a disabled feature is invoked."""

    pass


# New exception for missing configuration
class ConfigurationError(SettingsError):
    """Raised when a required configuration for a provider is missing."""

    pass


class InfiniteRedirectError(OrchestratorError):
    """Raised when a redirect loop is detected in pipeline execution."""

    pass


class InfiniteFallbackError(OrchestratorError):
    """Raised when a fallback loop is detected during execution."""

    pass


class PipelineContextInitializationError(OrchestratorError):
    """Raised when a typed pipeline context fails to initialize."""

    pass


class ContextInheritanceError(OrchestratorError):
    """Raised when inheriting context for a nested pipeline fails."""

    def __init__(
        self, missing_fields: list[str], parent_context_keys: list[str], child_model_name: str
    ) -> None:
        msg = (
            f"Failed to inherit context for {child_model_name}. Missing required fields: "
            f"{', '.join(missing_fields)}. Parent context provided: {', '.join(parent_context_keys)}."
        )
        super().__init__(msg)
        self.missing_fields = missing_fields
        self.parent_context_keys = parent_context_keys
        self.child_model_name = child_model_name


class UsageLimitExceededError(OrchestratorError):
    """Raised when a pipeline run exceeds its defined usage limits."""

    def __init__(self, message: str, result: "PipelineResult[Any]") -> None:
        super().__init__(message)
        self.result = result


class PipelineAbortSignal(Exception):
    """Special exception hooks can raise to stop a pipeline gracefully."""

    def __init__(self, message: str = "Pipeline aborted by hook.") -> None:
        super().__init__(message)


class PausedException(OrchestratorError):
    """Internal exception used to pause a pipeline."""

    def __init__(self, message: str = "Pipeline paused for human input.") -> None:
        super().__init__(message)


class ImproperStepInvocationError(OrchestratorError):
    """Raised when a ``Step`` object is invoked directly."""

    pass


class MissingAgentError(ConfigurationError):
    """Raised when a pipeline step is missing its agent."""

    pass


class TypeMismatchError(ConfigurationError):
    """Raised when consecutive steps have incompatible types."""

    pass


class AgentIOValidationError(OrchestratorError):
    """Raised when an agent's input or output validation fails."""

    pass


class FlujoFrameworkError(Exception):
    """Base exception for Flujo framework with enhanced error messages."""

    def __init__(
        self, message: str, suggestion: str | None = None, code: str | None = None
    ) -> None:
        self.message = message
        self.suggestion = suggestion
        self.code = code
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """
        Constructs a formatted error message for the exception.

        The message includes the main error description (`self.message`), and optionally
        appends a suggestion (`self.suggestion`) and an error code (`self.code`) if they
        are provided.

        Returns:
            str: A formatted string containing the error details.
        """
        msg = f"Flujo Error: {self.message}"
        if self.suggestion:
            msg += f"\n\nSuggestion: {self.suggestion}"
        if self.code:
            msg += f"\n\nError Code: {self.code}"
        return msg


class ContextFieldError(FlujoFrameworkError):
    """Raised when trying to set a field that doesn't exist in the context."""

    def __init__(
        self, field_name: str, context_class: str, available_fields: tuple[str, ...]
    ) -> None:
        super().__init__(
            f"'{context_class}' object has no field '{field_name}'",
            f"Available fields: {', '.join(available_fields)}",
            "CONTEXT_FIELD_ERROR",
        )


class StepInvocationError(FlujoFrameworkError):
    """Unified exception for step invocation errors.

    This exception replaces ImproperStepInvocationError for consistency
    and provides enhanced error messages for better debugging.
    """

    def __init__(self, step_name: str):
        super().__init__(
            f"Step '{step_name}' cannot be invoked directly",
            "Use Pipeline.from_step() or Step.solution() to wrap the step",
            "STEP_INVOCATION_ERROR",
        )


class ParallelStepError(FlujoFrameworkError):
    """Raised when there's an issue with parallel step execution."""

    def __init__(self, step_name: str, branch_name: str, issue: str):
        super().__init__(
            f"Parallel step '{step_name}' branch '{branch_name}': {issue}",
            "Consider using MergeStrategy.CONTEXT_UPDATE with field_mapping",
            "PARALLEL_STEP_ERROR",
        )
