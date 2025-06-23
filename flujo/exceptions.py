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


class PipelineContextInitializationError(OrchestratorError):
    """Raised when a typed pipeline context fails to initialize."""

    pass


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
