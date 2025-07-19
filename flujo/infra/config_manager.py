"""Configuration management for flujo with support for flujo.toml files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union
import tomllib
from pydantic import BaseModel

from .settings import Settings
from ..exceptions import ConfigurationError


class FlujoConfig(BaseModel):
    """Configuration loaded from flujo.toml files."""

    # CLI defaults
    solve: Optional[SolveConfig] = None
    bench: Optional[BenchConfig] = None
    run: Optional[RunConfig] = None

    # Global settings overrides
    settings: Optional[SettingsOverrides] = None

    # State backend configuration
    state_uri: Optional[str] = None


class SolveConfig(BaseModel):
    """Configuration for the solve command."""

    max_iters: Optional[int] = None
    k: Optional[int] = None
    reflection: Optional[bool] = None
    scorer: Optional[str] = None
    weights_path: Optional[str] = None
    solution_model: Optional[str] = None
    review_model: Optional[str] = None
    validator_model: Optional[str] = None
    reflection_model: Optional[str] = None


class BenchConfig(BaseModel):
    """Configuration for the bench command."""

    rounds: Optional[int] = None


class RunConfig(BaseModel):
    """Configuration for the run command."""

    pipeline_name: Optional[str] = None
    json_output: Optional[bool] = None


class SettingsOverrides(BaseModel):
    """Settings overrides from configuration file."""

    # Feature toggles
    reflection_enabled: Optional[bool] = None
    reward_enabled: Optional[bool] = None
    telemetry_export_enabled: Optional[bool] = None
    otlp_export_enabled: Optional[bool] = None

    # Default models
    default_solution_model: Optional[str] = None
    default_review_model: Optional[str] = None
    default_validator_model: Optional[str] = None
    default_reflection_model: Optional[str] = None
    default_self_improvement_model: Optional[str] = None
    default_repair_model: Optional[str] = None

    # Orchestrator tuning
    max_iters: Optional[int] = None
    k_variants: Optional[int] = None
    reflection_limit: Optional[int] = None
    scorer: Optional[str] = None
    t_schedule: Optional[list[float]] = None
    otlp_endpoint: Optional[str] = None
    agent_timeout: Optional[int] = None


class ConfigManager:
    """Manages configuration loading with proper precedence."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the configuration manager.

        Args:
            config_path: Path to the configuration file. If None, will search for flujo.toml
        """
        self.config_path = self._find_config_file(config_path)
        self._config: Optional[FlujoConfig] = None
        self._settings: Optional[Settings] = None

    def _find_config_file(self, config_path: Optional[Union[str, Path]]) -> Optional[Path]:
        """Find the configuration file to use."""
        if config_path:
            path = Path(config_path)
            if path.exists():
                return path
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        # Search for flujo.toml in current directory and parent directories
        current = Path.cwd()
        while current != current.parent:
            config_file = current / "flujo.toml"
            if config_file.exists():
                return config_file
            current = current.parent

        return None

    def load_config(self) -> FlujoConfig:
        """Load configuration from flujo.toml file."""
        if self._config is not None:
            return self._config

        if self.config_path is None:
            self._config = FlujoConfig()
            return self._config

        try:
            with open(self.config_path, "rb") as f:
                data = tomllib.load(f)

            # Extract configuration sections
            config_data = {}

            # CLI command configurations
            if "solve" in data:
                config_data["solve"] = data["solve"]
            if "bench" in data:
                config_data["bench"] = data["bench"]
            if "run" in data:
                config_data["run"] = data["run"]

            # Settings overrides
            if "settings" in data:
                config_data["settings"] = data["settings"]

            # State URI
            if "state_uri" in data:
                config_data["state_uri"] = data["state_uri"]

            self._config = FlujoConfig(**config_data)
            return self._config

        except FileNotFoundError as e:
            raise ConfigurationError(f"Configuration file not found at {self.config_path}: {e}")
        except PermissionError as e:
            raise ConfigurationError(f"Permission denied when accessing {self.config_path}: {e}")
        except tomllib.TOMLDecodeError as e:
            raise ConfigurationError(f"Failed to parse TOML configuration file {self.config_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"An unexpected error occurred while loading configuration from {self.config_path}: {e}")

    def get_settings(self) -> Settings:
        """Get settings with configuration file overrides applied."""
        if self._settings is not None:
            return self._settings

        # Start with the default settings using the proper constructor
        # BaseSettings handles the initialization automatically
        settings = Settings()
        config = self.load_config()

        # Apply settings overrides from configuration file
        if config.settings:
            for field_name, value in config.settings.model_dump(exclude_none=True).items():
                if hasattr(settings, field_name):
                    setattr(settings, field_name, value)

        self._settings = settings
        return self._settings

    def get_cli_defaults(self, command: str) -> Dict[str, Any]:
        """Get CLI defaults for a specific command."""
        config = self.load_config()

        if command == "solve" and config.solve:
            return config.solve.model_dump(exclude_none=True)
        elif command == "bench" and config.bench:
            return config.bench.model_dump(exclude_none=True)
        elif command == "run" and config.run:
            return config.run.model_dump(exclude_none=True)

        return {}

    def get_state_uri(self) -> Optional[str]:
        """Get the state URI from configuration."""
        config = self.load_config()
        return config.state_uri


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_settings() -> Settings:
    """Load settings with configuration file overrides."""
    return get_config_manager().get_settings()


def get_cli_defaults(command: str) -> Dict[str, Any]:
    """Get CLI defaults for a specific command."""
    return get_config_manager().get_cli_defaults(command)


def get_state_uri() -> Optional[str]:
    """Get the state URI from configuration."""
    return get_config_manager().get_state_uri()
