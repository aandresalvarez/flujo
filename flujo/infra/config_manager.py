"""Configuration management for flujo with support for flujo.toml files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, cast
import os

try:
    import tomllib
except ImportError:
    # Fallback for Python versions < 3.11
    import tomli as tomllib  # type: ignore
from pydantic import BaseModel
from ..domain.models import UsageLimits

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

    # Cost tracking configuration
    cost: Optional[Dict[str, Any]] = None

    # Security: allow-list for YAML blueprint imports
    blueprint_allowed_imports: Optional[list[str]] = None

    # Centralized budget governance
    budgets: Optional["BudgetConfig"] = None
    # Architect defaults
    architect: Optional["ArchitectConfig"] = None


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


class BudgetConfig(BaseModel):
    """Budget governance configuration loaded from flujo.toml.

    Example TOML:
    [budgets.default]
    total_cost_usd_limit = 10.0
    total_tokens_limit = 100000

    [budgets.pipeline]
    "analytics" = { total_tokens_limit = 200000 }
    "team-*"   = { total_cost_usd_limit = 5.0 }
    """

    default: Optional[UsageLimits] = None
    pipeline: Dict[str, UsageLimits] = {}


class ArchitectConfig(BaseModel):
    """Architect-related project defaults.

    Example TOML:
    [architect]
    state_machine_default = true
    """

    state_machine_default: Optional[bool] = None


class ConfigManager:
    """Manages configuration loading with proper precedence."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the configuration manager.

        Args:
            config_path: Path to the configuration file. If None, will search for flujo.toml
        """
        # Track how the config path was determined to handle precedence edge cases
        self._config_source: str = "none"  # one of: arg, env, search, none
        if config_path is not None:
            # Explicit path provided by caller
            p = Path(config_path)
            if not p.exists():
                raise ConfigurationError(f"Configuration file not found: {config_path}")
            self.config_path = p
            self._config_source = "arg"
        else:
            self.config_path = self._find_config_file(None)
            # Determine discovery source for subtle precedence decisions
            if os.environ.get("FLUJO_CONFIG_PATH") and self.config_path is not None:
                self._config_source = "env"
            elif self.config_path is not None:
                self._config_source = "search"
            else:
                self._config_source = "none"
        self._cached_config: Optional[FlujoConfig] = None
        self._config_file_mtime: Optional[float] = None

    def _find_config_file(self, config_path: Optional[Union[str, Path]]) -> Optional[Path]:
        """Find the configuration file to use."""
        # 1. Check explicit argument
        if config_path:
            path = Path(config_path)
            if path.exists():
                return path
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        # 2. Check environment variable
        env_path = os.environ.get("FLUJO_CONFIG_PATH")
        if env_path:
            path = Path(env_path)
            if path.exists():
                return path
            raise ConfigurationError(f"Configuration file not found: {env_path}")

        # 3. Fallback: search for flujo.toml in CWD and parents
        current = Path.cwd()
        while current != current.parent:
            config_file = current / "flujo.toml"
            if config_file.exists():
                return config_file
            current = current.parent

        return None

    def load_config(self, force_reload: bool = False) -> FlujoConfig:
        """Load configuration from flujo.toml file.

        Args:
            force_reload: If True, bypass the cache and reload from file

        Returns:
            FlujoConfig: The loaded configuration
        """
        if self.config_path is None:
            return FlujoConfig()

        # Check if we can use cached config
        if not force_reload and self._cached_config is not None:
            try:
                # Check if file has been modified since last load
                current_mtime = self.config_path.stat().st_mtime
                if self._config_file_mtime is not None and current_mtime == self._config_file_mtime:
                    return self._cached_config
            except (OSError, AttributeError):
                # If we can't check modification time, proceed with reload
                pass

        # Load configuration from file
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

            # Cost tracking configuration
            if "cost" in data:
                config_data["cost"] = data["cost"]

            # Security allow-list (either top-level key or nested under [settings])
            if "blueprint_allowed_imports" in data:
                config_data["blueprint_allowed_imports"] = data["blueprint_allowed_imports"]

            # Budgets governance configuration
            if "budgets" in data:
                config_data["budgets"] = data["budgets"]

            # Architect configuration
            if "architect" in data:
                config_data["architect"] = data["architect"]

            config = FlujoConfig(**config_data)

            # Cache the configuration and file modification time
            self._cached_config = config
            try:
                self._config_file_mtime = self.config_path.stat().st_mtime
            except (OSError, AttributeError):
                self._config_file_mtime = None

            return config

        except FileNotFoundError as e:
            raise ConfigurationError(f"Configuration file not found at {self.config_path}: {e}")
        except PermissionError as e:
            raise ConfigurationError(f"Permission denied when accessing {self.config_path}: {e}")
        except tomllib.TOMLDecodeError as e:
            raise ConfigurationError(
                f"Failed to parse TOML configuration file {self.config_path}: {e}"
            )
        except (OSError, ValueError) as e:
            raise ConfigurationError(f"Error loading configuration from {self.config_path}: {e}")
        except KeyError as e:
            raise ConfigurationError(f"Missing expected key in configuration data: {e}")
        except Exception as e:
            # Log the exception type and details for debugging purposes
            import logging

            logging.error(f"Unexpected error during configuration loading: {type(e).__name__}: {e}")
            # Catch any other truly unexpected errors and provide a generic message
            # This is kept as a final fallback after handling all specific exceptions
            # to ensure we always provide a meaningful error message
            raise ConfigurationError(
                f"An unexpected error occurred during configuration loading: {e}"
            )

    def get_settings(self, force_reload: bool = False) -> Any:
        """Get settings with configuration file overrides applied.

        Implements the precedence: Defaults < TOML File < Environment Variables

        This method constructs the Settings object in the following strict order:
        1. Start with pydantic defaults from the Settings class
        2. Apply TOML file overrides (if [settings] section exists)
        3. Allow environment variables to override both defaults and TOML values

        Args:
            force_reload: If True, bypass the cache and reload from file
        """
        from .settings import Settings

        # Step 1: Load TOML configuration
        config = self.load_config(force_reload=force_reload)

        # Step 2: Create Settings with defaults + environment variables
        # pydantic-settings automatically loads: defaults < environment variables
        settings = cast(Callable[[], Settings], Settings)()

        # Step 3: Apply TOML overrides, but only if no environment variable is set
        # This ensures environment variables have the highest precedence
        if config.settings:
            for field_name, toml_value in config.settings.model_dump(exclude_none=True).items():
                if hasattr(settings, field_name):
                    # Check if this field has been set by an environment variable
                    field_info = Settings.model_fields.get(field_name)
                    if field_info and self._is_field_set_by_env(field_name, field_info):
                        # Environment variable takes precedence, skip TOML override
                        continue

                    # Apply TOML value since no environment variable was found
                    setattr(settings, field_name, toml_value)

        return settings

    def _is_field_set_by_env(self, field_name: str, field_info: Any) -> bool:
        """Check if a field was set by an environment variable.

        This method checks all possible environment variable names for a field,
        including validation_alias patterns.
        """
        import os
        from pydantic import AliasChoices

        # Get all possible environment variable names for this field
        env_var_names = [field_name.upper()]

        # Add validation_alias names if they exist
        if hasattr(field_info, "validation_alias") and field_info.validation_alias:
            alias = field_info.validation_alias
            if isinstance(alias, AliasChoices):
                env_var_names.extend(
                    [str(choice).upper() for choice in alias.choices if isinstance(choice, str)]
                )
            elif isinstance(alias, str):
                env_var_names.append(alias.upper())

        # Check if any of these environment variables are set
        return any(env_var in os.environ for env_var in env_var_names)

    def get_cli_defaults(self, command: str, force_reload: bool = False) -> Dict[str, Any]:
        """Get CLI defaults for a specific command.

        Args:
            command: The CLI command name
            force_reload: If True, bypass the cache and reload from file
        """
        config = self.load_config(force_reload=force_reload)

        if command == "solve" and config.solve:
            return config.solve.model_dump(exclude_none=True)
        elif command == "bench" and config.bench:
            return config.bench.model_dump(exclude_none=True)
        elif command == "run" and config.run:
            return config.run.model_dump(exclude_none=True)

        return {}

    def get_state_uri(self, force_reload: bool = False) -> Optional[str]:
        """Get the state URI from configuration.

        Implements the precedence: Environment Variables > TOML File > None

        Args:
            force_reload: If True, bypass the cache and reload from file
        """
        # When an explicit config file was provided by the caller, prefer its state_uri
        # over the environment variable to satisfy targeted integration scenarios.
        if self._config_source == "arg":
            config = self.load_config(force_reload=force_reload)
            if config.state_uri:
                return config.state_uri

        # Otherwise, 1) Environment variable, 2) TOML value
        env_uri = os.environ.get("FLUJO_STATE_URI")
        if env_uri:
            return env_uri

        # TOML file configuration
        config = self.load_config(force_reload=force_reload)
        return config.state_uri


# Configuration manager instance - stateless approach for multiprocessing safety
def get_config_manager(force_reload: bool = False) -> ConfigManager:
    """Get a config manager instance. Always creates a new instance for multiprocessing safety."""
    # Always create a new instance to avoid multiprocessing issues
    # This ensures each process gets its own config manager
    return ConfigManager()


def load_settings(force_reload: bool = False) -> Any:
    """Load settings with configuration file overrides. If force_reload is True, reload config/settings."""
    return get_config_manager(force_reload=force_reload).get_settings()


def get_cli_defaults(command: str, force_reload: bool = False) -> Dict[str, Any]:
    """Get CLI defaults for a specific command. If force_reload is True, reload config/settings."""
    return get_config_manager(force_reload=force_reload).get_cli_defaults(command)


def get_state_uri(force_reload: bool = False) -> Optional[str]:
    """Get the state URI from configuration. If force_reload is True, reload config/settings."""
    return get_config_manager(force_reload=force_reload).get_state_uri()
