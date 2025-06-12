"""Custom exceptions for the orchestrator."""

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