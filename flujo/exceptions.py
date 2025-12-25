"""Custom exceptions for the orchestrator."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from flujo.domain.models import PipelineResult


# ============================================================================
# Unified Exception Hierarchy (Phase 1)
# ============================================================================


class FlujoError(Exception):
    """Unified base exception for all Flujo errors.

    This is the new base class for all Flujo exceptions. It provides
    enhanced error messages with suggestions and error codes.

    Examples:
        >>> raise FlujoError("Something went wrong", suggestion="Try X", code="E001")
        >>> raise ConfigurationError("Missing config", suggestion="Set FLUJO_API_KEY")
    """

    def __init__(
        self,
        message: str,
        *,
        suggestion: str | None = None,
        code: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.message = message
        self.suggestion = suggestion
        self.code = code
        super().__init__(self._format_message())
        if cause:
            self.__cause__ = cause

    def _format_message(self) -> str:
        """Format the error message with optional suggestion and code.

        For backward compatibility: only format if suggestion or code is provided.
        Otherwise, return the plain message.
        """
        # Handle empty messages (backward compatibility for control flow exceptions)
        if not self.message.strip():
            # For empty messages, return empty string if no code/suggestion
            if not self.suggestion and not self.code:
                return ""
            # If there's a code or suggestion, format minimally
            parts = []
            if self.code:
                parts.append(f"Error Code: {self.code}")
            if self.suggestion:
                parts.append(self.suggestion)
            return "\n".join(parts) if parts else ""

        # Backward compatibility: only format if suggestion or code is explicitly provided
        # This preserves the original message format for existing code
        if not self.suggestion and not self.code:
            return self.message

        # Format with [Flujo] prefix when suggestion/code are provided
        parts = [f"[Flujo] {self.message}"]
        if self.suggestion:
            parts.append(f"\n  Suggestion: {self.suggestion}")
        if self.code:
            parts.append(f"\n  Error Code: {self.code}")
        return "".join(parts)


# Category exception classes
class ConfigurationError(FlujoError):
    """Errors related to configuration and setup."""

    pass


class ExecutionError(FlujoError):
    """Errors occurring during pipeline execution."""

    pass


class ControlFlowError(FlujoError):
    """Non-retryable control flow signals (NEVER catch and swallow).

    These exceptions are used for orchestration and should always be
    re-raised to let the runner handle workflow control.
    """

    pass


class ContextError(FlujoError):
    """Errors related to context management."""

    pass


class ValidationError(FlujoError):
    """Errors related to validation (types, templates, schemas)."""

    pass


class SettingsError(ConfigurationError):
    """Raised for configuration-related errors."""

    pass


class OrchestratorRetryError(ExecutionError):
    """Raised when an agent operation fails after all retries."""

    pass


class RewardModelUnavailable(ConfigurationError):
    """Raised when the reward model is required but unavailable."""

    pass


class FeatureDisabled(ConfigurationError):
    """Raised when a disabled feature is invoked."""

    pass


# Note: ConfigurationError is now defined above as a category class
# This was previously inheriting from SettingsError, but now SettingsError inherits from ConfigurationError


class PricingNotConfiguredError(ConfigurationError):
    """Raised when strict pricing mode is enabled but pricing is not configured for a model."""

    def __init__(self, provider: str | None, model: str) -> None:
        provider_str = f"'{provider}'" if provider else "None"
        message = (
            f"Strict pricing is enabled, but no configuration was found for "
            f"provider={provider_str}, model='{model}' in flujo.toml."
        )
        super().__init__(message)
        self.provider = provider
        self.model = model


class InfiniteRedirectError(ControlFlowError):
    """Raised when a redirect loop is detected in pipeline execution.

    This is a control flow exception and should NEVER be caught and
    converted to a StepResult. Always re-raise it.
    """

    pass


class InfiniteFallbackError(ControlFlowError):
    """Raised when a fallback loop is detected during execution.

    This is a control flow exception and should NEVER be caught and
    converted to a StepResult. Always re-raise it.
    """

    pass


class PipelineContextInitializationError(ContextError):
    """Raised when a typed pipeline context fails to initialize."""

    pass


class ContextInheritanceError(ContextError):
    """Raised when inheriting context for a nested pipeline fails."""

    def __init__(
        self,
        missing_fields: list[str],
        parent_context_keys: list[str],
        child_model_name: str,
    ) -> None:
        msg = (
            f"Context Inheritance Error: Failed to inherit context for {child_model_name}. "
            f"Missing required fields: {', '.join(missing_fields)}. "
            f"Parent context provided: {', '.join(parent_context_keys)}."
        )
        super().__init__(
            msg,
            suggestion=f"Ensure parent context includes fields: {', '.join(missing_fields)}",
            code="CONTEXT_INHERITANCE_ERROR",
        )
        self.missing_fields = missing_fields
        self.parent_context_keys = parent_context_keys
        self.child_model_name = child_model_name


class UsageLimitExceededError(ExecutionError):
    """Raised when a pipeline run exceeds its defined usage limits."""

    def __init__(self, message: str, result: Optional["PipelineResult[Any]"] = None) -> None:
        # Don't add formatting for backward compatibility with existing tests
        # Tests expect plain messages like "Cost limit of $1 exceeded"
        super().__init__(message)
        self.result = result


class PipelineAbortSignal(ControlFlowError):
    """Special exception hooks can raise to stop a pipeline gracefully.

    This is a control flow exception and should NEVER be caught and
    converted to a StepResult. Always re-raise it.
    """

    def __init__(self, message: str = "Pipeline aborted by hook.") -> None:
        super().__init__(message, code="PIPELINE_ABORT")


class PausedException(ControlFlowError):
    """Internal exception used to pause a pipeline.

    This is a control flow exception and should NEVER be caught and
    converted to a StepResult. Always re-raise it.
    """

    def __init__(self, message: str = "Pipeline paused for human input.") -> None:
        # For backward compatibility: if message is empty, don't add code formatting
        if not message or not message.strip():
            super().__init__(message)
        else:
            super().__init__(message, code="PAUSED")


class ImproperStepInvocationError(ExecutionError):
    """DEPRECATED: Raised when a ``Step`` object is invoked directly.

    .. deprecated:: 0.4.0
        Use :class:`StepInvocationError` instead.
        Will be removed in v1.0.0.
    """

    pass


class MissingAgentError(ConfigurationError):
    """Raised when a pipeline step is missing its agent."""

    pass


class TypeMismatchError(ValidationError):
    """Raised when consecutive steps have incompatible types."""

    pass


class AgentIOValidationError(ExecutionError):
    """Raised when an agent's input or output validation fails."""

    pass


class NonRetryableError(ExecutionError):
    """Base class for errors that should not be retried in the pipeline."""

    pass


class HitlPolicyError(ConfigurationError):
    """Raised when HITL steps are disabled by policy or configuration."""

    pass


class ResumeError(ExecutionError):
    """Raised when a pipeline cannot be resumed due to invalid state."""

    pass


class ReplayError(ExecutionError):
    """Raised when a replay attempt cannot proceed."""

    pass


class PluginError(ExecutionError):
    """Raised when a plugin execution fails."""

    pass


class ContextFieldError(ContextError):
    """Raised when trying to set a field that doesn't exist in the context."""

    def __init__(self, field_name: str, context_class: str, available_fields: list[str]) -> None:
        super().__init__(
            f"'{context_class}' object has no field '{field_name}'",
            suggestion=f"Available fields: {', '.join(available_fields)}",
            code="CONTEXT_FIELD_ERROR",
        )


class StepInvocationError(ExecutionError):
    """Unified exception for step invocation errors.

    This exception replaces ImproperStepInvocationError for consistency
    and provides enhanced error messages for better debugging.

    Note: ImproperStepInvocationError is deprecated and will be removed in a future version.
    Use StepInvocationError for new code.
    """

    def __init__(self, step_name: str) -> None:
        super().__init__(
            f"Step '{step_name}' cannot be invoked directly",
            suggestion="Use Pipeline.from_step() or Step.solution() to wrap the step",
            code="STEP_INVOCATION_ERROR",
        )


class ParallelStepError(ExecutionError):
    """Raised when there's an issue with parallel step execution."""

    def __init__(self, step_name: str, branch_name: str, issue: str) -> None:
        super().__init__(
            f"Parallel step '{step_name}' branch '{branch_name}': {issue}",
            suggestion="Consider using MergeStrategy.CONTEXT_UPDATE with field_mapping",
            code="PARALLEL_STEP_ERROR",
        )


class TemplateResolutionError(ValidationError):
    """Raised when a template variable cannot be resolved (strict mode)."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            suggestion="Use '{{ previous_step.field }}' or '{{ steps.step_name.output.field }}' for explicit references",
            code="TEMPLATE_RESOLUTION_ERROR",
        )


# ============================================================================
# Additional Context Errors
# ============================================================================


class ContextIsolationError(ContextError):
    """Raised when context isolation fails or is violated."""

    pass


class ContextMergeError(ContextError):
    """Raised when context merging fails."""

    pass


class ContextMutationError(ContextError):
    """Raised when a context mutation violation is detected (strict mode)."""

    pass
