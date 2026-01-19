"""Configuration management for flujo with support for flujo.toml files."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Optional, Union
import os
import os as _os
import tomllib
from pydantic import BaseModel, Field
from ..domain.models import UsageLimits

from ..exceptions import ConfigurationError
from ..type_definitions.common import JSONObject


class ArosConfig(BaseModel):
    """AROS defaults loaded from flujo.toml [aros] section.

    These values are accessed via ConfigManager; do not read TOML directly.
    """

    enabled: bool = True
    structured_output_default: str = "off"  # off | auto | openai_json | outlines | xgrammar
    enable_aop_default: str = "off"  # off | minimal | full
    coercion_tolerant_level: int = 0  # 0=off, 1=json5, 2=json-repair
    max_unescape_depth: int = 2  # Stage 0b depth cap
    anyof_strategy: str = "first-pass"  # branch chooser strategy
    enable_reasoning_precheck: bool = False


class TemplateConfig(BaseModel):
    """Template rendering configuration loaded from flujo.toml [template] section.

    Controls how templates handle undefined variables and logging behavior.
    """

    undefined_variables: str = (
        "warn"  # "strict" (raise error), "warn" (log warning), "ignore" (silent)
    )
    log_resolution: bool = False  # Log template resolution process in debug mode


class SandboxOverrides(BaseModel):
    """Sandbox settings overrides from flujo.toml [settings.sandbox]."""

    mode: Optional[str] = None
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: Optional[int] = None
    verify_ssl: Optional[bool] = None
    docker_image: Optional[str] = None
    docker_pull: Optional[bool] = None
    docker_mem_limit: Optional[str] = None
    docker_pids_limit: Optional[int] = None
    docker_network_mode: Optional[str] = None


class FlujoConfig(BaseModel):
    """Configuration loaded from flujo.toml files."""

    # CLI defaults
    solve: Optional[SolveConfig] = None
    bench: Optional[BenchConfig] = None
    run: Optional[RunConfig] = None

    # Global settings overrides
    settings: Optional[SettingsOverrides] = None

    # Optional path to a dotenv env file relative to the project root (config file directory)
    env_file: Optional[str] = None

    # State backend configuration
    state_uri: Optional[str] = None

    # Cost tracking configuration
    cost: Optional[JSONObject] = None

    # Security: allow-list for YAML blueprint imports
    blueprint_allowed_imports: Optional[list[str]] = None

    # Centralized budget governance
    budgets: Optional["BudgetConfig"] = None
    # Architect defaults
    architect: Optional["ArchitectConfig"] = None

    # AROS defaults
    aros: Optional[ArosConfig] = None
    # Validation configuration
    validation: Optional["ValidationConfig"] = None
    # Template configuration
    template: Optional[TemplateConfig] = None
    # Lockfile configuration
    lockfile: Optional[LockfileConfig] = None

    # Governance policy module (shortcut at top level)
    governance_policy_module: Optional[str] = None


class ValidationConfig(BaseModel):
    """Validation settings including named rule profiles.

    Example TOML:
    [validation.profiles.strict]
    V-T* = "error"
    V-A5 = "warning"
    """

    profiles: Optional[dict[str, JSONObject]] = None


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
    state_backend_span_export_enabled: Optional[bool] = None
    memory_indexing_enabled: Optional[bool] = None
    memory_embedding_model: Optional[str] = None
    memory_embedding_dimensions: Optional[int] = None

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

    # Postgres backend tuning
    postgres_pool_min: Optional[int] = None
    postgres_pool_max: Optional[int] = None

    # Template filter controls
    enabled_template_filters: Optional[list[str]] = None

    # Governance and shadow eval overrides
    governance_mode: Optional[str] = None
    governance_policy_module: Optional[str] = None
    governance_tool_allowlist: Optional[list[str]] = None
    shadow_eval_sink: Optional[str] = None
    sandbox: Optional[SandboxOverrides] = None


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
    pipeline: dict[str, UsageLimits] = Field(default_factory=dict)


class LockfileConfig(BaseModel):
    """Lockfile configuration loaded from flujo.toml [lockfile] section.

    Example TOML:
    [lockfile]
    enabled = true
    external_files = ["data/*.json", "schemas/*.yaml"]
    include_model_info = true
    include_tool_schemas = false
    auto_generate_on_run = true
    """

    enabled: bool = True
    external_files: Optional[list[str]] = None
    include_model_info: bool = True
    include_tool_schemas: bool = False
    auto_generate_on_run: bool = True


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
        # Attribute annotation ensures mypy understands Optional type across branches
        self.config_path: Optional[Path]
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
        self._env_signature: Optional[str] = None

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

    def _validate_structures(self, data: dict[str, Any]) -> None:
        """Fail fast on obviously invalid config shapes before model parsing."""
        errors: list[str] = []

        def _expect_mapping(key: str) -> None:
            value = data.get(key)
            if value is None:
                return
            if not isinstance(value, dict):
                errors.append(f"[{key}] must be a table/dict, got {type(value).__name__}")

        def _expect_list_of_str(key: str) -> None:
            value = data.get(key)
            if value is None:
                return
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                errors.append(f"[{key}] must be a list of strings")

        _expect_mapping("solve")
        _expect_mapping("bench")
        _expect_mapping("run")
        _expect_mapping("settings")
        _expect_mapping("cost")
        _expect_mapping("budgets")
        _expect_mapping("architect")
        _expect_mapping("validation")
        _expect_mapping("template")
        _expect_mapping("lockfile")
        _expect_list_of_str("blueprint_allowed_imports")

        if errors:
            raise ConfigurationError("Invalid configuration structure: " + "; ".join(errors))

    def load_config(self, force_reload: bool = False) -> FlujoConfig:
        """Load configuration from flujo.toml file.

        Args:
            force_reload: If True, bypass the cache and reload from file

        Returns:
            FlujoConfig: The loaded configuration
        """

        def _parse_allowed_imports_env(val: Optional[str]) -> Optional[list[str]]:
            """Parse FLUJO_BLUEPRINT_ALLOWED_IMPORTS into a normalized list of prefixes."""
            if val is None:
                return None
            raw = val.strip()
            if raw == "":
                return None
            # Support wildcard allow-all and comma/semicolon separated lists
            if raw == "*":
                return ["*"]
            tokens: list[str] = []
            for part in raw.replace(";", ",").split(","):
                tok = part.strip()
                if tok:
                    tokens.append(tok)
            return tokens or None

        env_allowed = _parse_allowed_imports_env(os.environ.get("FLUJO_BLUEPRINT_ALLOWED_IMPORTS"))
        test_mode_enabled = str(os.environ.get("FLUJO_TEST_MODE", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        if self.config_path is None:
            current_env_sig = self._compute_env_signature()
            if (
                not force_reload
                and self._cached_config is not None
                and self._env_signature == current_env_sig
            ):
                return self._cached_config
            config_data_env: JSONObject = {}
            if env_allowed is not None:
                config_data_env["blueprint_allowed_imports"] = env_allowed
            elif test_mode_enabled and "blueprint_allowed_imports" not in config_data_env:
                # Default allow-list for test environments to unblock fixture imports
                config_data_env["blueprint_allowed_imports"] = ["tests", "skills"]
            self._cached_config = FlujoConfig(**config_data_env)
            self._config_file_mtime = None
            self._env_signature = current_env_sig
            return self._cached_config

        # Check if we can use cached config
        if not force_reload and self._cached_config is not None:
            try:
                # Check if file has been modified since last load or env changed
                current_mtime = self.config_path.stat().st_mtime
                current_env_sig = self._compute_env_signature()
                if (
                    self._config_file_mtime is not None
                    and current_mtime == self._config_file_mtime
                    and self._env_signature == current_env_sig
                ):
                    return self._cached_config
            except (OSError, AttributeError):
                # If we can't check modification time, proceed with reload
                pass

        # Load configuration from file
        try:
            with open(self.config_path, "rb") as f:
                data = tomllib.load(f)

            self._validate_structures(data)

            # Extract configuration sections
            config_data: JSONObject = {}

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

            # Compatibility: map [blueprint].enabled_filters -> settings.enabled_template_filters
            try:
                blueprint = data.get("blueprint")
                if isinstance(blueprint, dict) and "enabled_filters" in blueprint:
                    cfg_settings = dict(config_data.get("settings", {}))
                    cfg_settings["enabled_template_filters"] = blueprint.get("enabled_filters")
                    config_data["settings"] = cfg_settings
            except Exception:
                pass

            # Optional dotenv file path for API keys and secrets
            if "env_file" in data:
                config_data["env_file"] = data["env_file"]

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

            # Validation configuration
            if "validation" in data:
                config_data["validation"] = data["validation"]

            # Template configuration
            if "template" in data:
                config_data["template"] = data["template"]

            # Lockfile configuration
            if "lockfile" in data:
                config_data["lockfile"] = data["lockfile"]

            # Governance policy module shortcut
            if "governance_policy_module" in data:
                cfg_settings = dict(config_data.get("settings", {}))
                cfg_settings["governance_policy_module"] = data["governance_policy_module"]
                config_data["settings"] = cfg_settings

            # Environment override for blueprint imports (highest precedence)
            if env_allowed is not None:
                config_data["blueprint_allowed_imports"] = env_allowed
            elif test_mode_enabled and "blueprint_allowed_imports" not in config_data:
                config_data["blueprint_allowed_imports"] = ["tests", "skills"]

            # Environment override for template filters (env > file)
            try:
                env_filters = os.environ.get("FLUJO_ENABLED_FILTERS")
                if env_filters:
                    parts = [p.strip() for p in env_filters.split(",") if p.strip()]
                    if parts:
                        cfg_settings = dict(config_data.get("settings", {}))
                        cfg_settings["enabled_template_filters"] = parts
                        config_data["settings"] = cfg_settings
            except Exception:
                pass

            config = FlujoConfig(**config_data)

            # Cache the configuration and file modification time
            self._cached_config = config
            try:
                self._config_file_mtime = self.config_path.stat().st_mtime
            except (OSError, AttributeError):
                self._config_file_mtime = None
            self._env_signature = self._compute_env_signature()

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

    def _compute_env_signature(self) -> str:
        """Compute a stable signature of env vars that influence settings."""
        relevant: dict[str, str] = {}
        try:
            for k, v in os.environ.items():
                if k.startswith("FLUJO_") or k in {
                    "OPENAI_API_KEY",
                    "ANTHROPIC_API_KEY",
                    "AZURE_OPENAI_API_KEY",
                    "HF_TOKEN",
                    "PINECONE_API_KEY",
                    "REDIS_URL",
                    "DATABASE_URL",
                }:
                    relevant[k] = v
        except Exception:
            relevant = {}
        items = [f"{k}={relevant[k]}" for k in sorted(relevant)]
        return "|".join(items)

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

        # If an env file is declared in flujo.toml, load it before constructing Settings
        try:
            config_for_env = self.load_config(force_reload=force_reload)
            env_file = getattr(config_for_env, "env_file", None)
            if env_file:
                # Resolve relative to the directory of flujo.toml
                base_dir = self.config_path.parent if self.config_path else Path.cwd()
                env_path = Path(env_file)
                if not env_path.is_absolute():
                    env_path = (base_dir / env_path).resolve()
                # Load dotenv file (non-fatal if missing)
                try:
                    import dotenv as _dotenv

                    _dotenv.load_dotenv(env_path.as_posix(), override=False)
                except Exception:
                    pass
        except Exception:
            # Do not fail settings load due to env file issues
            pass

        # Step 1: Load TOML configuration
        config = self.load_config(force_reload=force_reload)

        # Step 2: Create Settings with defaults + environment variables
        # pydantic-settings automatically loads: defaults < environment variables
        settings = Settings()
        self._apply_sandbox_env_overrides(settings)

        # Step 3: Apply TOML overrides, but only if no environment variable is set
        # This ensures environment variables have the highest precedence
        if config.settings:
            for field_name, toml_value in config.settings.model_dump(exclude_none=True).items():
                if field_name == "sandbox" and isinstance(toml_value, dict):
                    sandbox_obj = getattr(settings, "sandbox", None)
                    if sandbox_obj is not None:
                        sandbox_fields = getattr(type(sandbox_obj), "model_fields", {}) or {}
                        for sandbox_key, sandbox_value in toml_value.items():
                            if sandbox_value is None:
                                continue
                            if not hasattr(sandbox_obj, sandbox_key):
                                continue
                            field_info = sandbox_fields.get(sandbox_key)
                            if field_info and self._is_field_set_by_env(sandbox_key, field_info):
                                continue
                            try:
                                setattr(sandbox_obj, sandbox_key, sandbox_value)
                            except Exception:
                                pass
                    continue
                if hasattr(settings, field_name):
                    # Check if this field has been set by an environment variable
                    field_info = Settings.model_fields.get(field_name)
                    if field_info and self._is_field_set_by_env(field_name, field_info):
                        # Environment variable takes precedence, skip TOML override
                        continue

                    # Normalize list-based TOML overrides to match Settings types.
                    if field_name == "governance_tool_allowlist" and isinstance(toml_value, list):
                        toml_value = ",".join(str(v).strip() for v in toml_value if str(v).strip())

                    # Apply TOML value since no environment variable was found
                    setattr(settings, field_name, toml_value)

        return settings

    def get_aros_config(self, force_reload: bool = False) -> ArosConfig:
        """Return AROS defaults from flujo.toml with safe fallbacks.

        This does not read environment variables directly and provides
        conservative defaults when the section is absent.
        """
        try:
            cfg = self.load_config(force_reload=force_reload)
            if cfg.aros and isinstance(cfg.aros, ArosConfig):
                return cfg.aros
        except Exception:
            pass
        return ArosConfig()  # defaults

    def _get_env_value_for_field(self, field_name: str, field_info: Any) -> Optional[str]:
        """Return the environment variable value for a field when set.

        Values are read from ``os.environ`` and are therefore always strings;
        this method returns the matching string value or ``None`` if no
        relevant environment variable is defined.
        """
        from pydantic import AliasChoices

        env_var_names = [field_name.upper()]
        if hasattr(field_info, "validation_alias") and field_info.validation_alias:
            alias = field_info.validation_alias
            if isinstance(alias, AliasChoices):
                for choice in alias.choices:
                    if isinstance(choice, str):
                        env_var_names.append(choice.upper())
                        env_var_names.append(choice)
            elif isinstance(alias, str):
                env_var_names.append(alias.upper())
                env_var_names.append(alias)

        for env_var in env_var_names:
            if env_var in os.environ:
                return os.environ.get(env_var)
        return None

    def _is_field_set_by_env(self, field_name: str, field_info: Any) -> bool:
        """Check if a field was set by an environment variable."""
        return self._get_env_value_for_field(field_name, field_info) is not None

    def _apply_sandbox_env_overrides(self, settings: Any) -> None:
        """Apply env overrides to nested sandbox settings."""
        sandbox_obj = getattr(settings, "sandbox", None)
        if sandbox_obj is None:
            return
        try:
            from .settings import SandboxSettings

            sandbox_fields = SandboxSettings.model_fields
        except Exception:
            return

        sandbox_data = {}
        try:
            sandbox_data = sandbox_obj.model_dump()
        except Exception:
            pass
        env_overrides: dict[str, str] = {}
        for field_name, field_info in sandbox_fields.items():
            env_value = self._get_env_value_for_field(field_name, field_info)
            if env_value is not None:
                env_overrides[field_name] = env_value
        if not env_overrides:
            return
        try:
            merged = {**sandbox_data, **env_overrides}
            setattr(settings, "sandbox", SandboxSettings.model_validate(merged))
        except Exception:
            for key, value in env_overrides.items():
                try:
                    setattr(sandbox_obj, key, value)
                except Exception:
                    pass

    def get_cli_defaults(self, command: str, force_reload: bool = False) -> JSONObject:
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
        elif command == "lock" and config.lockfile:
            return config.lockfile.model_dump(exclude_none=True)

        return {}

    def get_state_uri(self, force_reload: bool = False) -> Optional[str]:
        """Get the state URI from configuration.

        Implements the precedence: Environment Variables > TOML File > None,
        except when this manager was initialized with an explicit config path
        (source = "arg"). In that case, prefer the TOML value to honor
        the caller’s explicit configuration file selection and avoid test
        contamination from environment variables set by other suites.

        Args:
            force_reload: If True, bypass the cache and reload from file
        """
        # Precedence: 1) Environment variable, 2) TOML value
        env_uri = os.environ.get("FLUJO_STATE_URI")
        if env_uri:
            return env_uri

        # Test mode isolation: when FLUJO_TEST_MODE=1 and FLUJO_TEST_STATE_DIR is set,
        # use an isolated SQLite database in that directory for test isolation.
        test_mode = str(os.environ.get("FLUJO_TEST_MODE", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        test_state_dir = os.environ.get("FLUJO_TEST_STATE_DIR", "").strip()
        if test_mode and test_state_dir:
            from pathlib import Path

            test_db_path = Path(test_state_dir) / "flujo_ops.db"
            return f"sqlite:///{test_db_path.as_posix()}"

        # TOML file configuration
        config = self.load_config(force_reload=force_reload)
        return config.state_uri


_CONFIG_MANAGER_LOCK = threading.Lock()
_CONFIG_MANAGER_CACHE: dict[int, ConfigManager] = {}


def _reset_config_cache_and_lock() -> None:
    """Reinitialize lock/cache after a fork to avoid inherited locked state."""
    global _CONFIG_MANAGER_LOCK
    _CONFIG_MANAGER_LOCK = threading.Lock()
    _CONFIG_MANAGER_CACHE.clear()


# Protect prefork servers (gunicorn/uvicorn with workers) from inheriting a locked mutex.
if hasattr(_os, "register_at_fork"):
    try:
        _os.register_at_fork(after_in_child=_reset_config_cache_and_lock)
    except Exception:
        # If registration fails, we still proceed; worst case we fall back to runtime checks.
        pass


def get_config_manager(force_reload: bool = False) -> ConfigManager:
    """Get a process-local ConfigManager instance with optional cache refresh."""
    pid = os.getpid()
    current_env_path = os.environ.get("FLUJO_CONFIG_PATH")

    def _needs_refresh(cached: ConfigManager) -> bool:
        """Refresh when env-provided config path changes or is cleared."""
        if current_env_path:
            env_path = Path(current_env_path)
            if cached._config_source != "env":
                return True
            return cached.config_path != env_path
        # No env path set: drop cached env-sourced manager to allow search/defaults
        return cached._config_source == "env"

    with _CONFIG_MANAGER_LOCK:
        cached = _CONFIG_MANAGER_CACHE.get(pid)
        if force_reload or cached is None or _needs_refresh(cached):
            _CONFIG_MANAGER_CACHE[pid] = ConfigManager()
        return _CONFIG_MANAGER_CACHE[pid]


def invalidate_config_cache() -> None:
    """Clear the process-local ConfigManager cache (useful in tests)."""
    with _CONFIG_MANAGER_LOCK:
        _CONFIG_MANAGER_CACHE.clear()


def load_settings(force_reload: bool = False) -> Any:
    """Load settings with configuration file overrides. If force_reload is True, reload config/settings."""
    return get_config_manager(force_reload=force_reload).get_settings()


def get_cli_defaults(command: str, force_reload: bool = False) -> JSONObject:
    """Get CLI defaults for a specific command. If force_reload is True, reload config/settings."""
    return get_config_manager(force_reload=force_reload).get_cli_defaults(command)


def get_state_uri(force_reload: bool = False) -> Optional[str]:
    """Get the state URI from configuration. If force_reload is True, reload config/settings."""
    return get_config_manager(force_reload=force_reload).get_state_uri()


def get_aros_config(force_reload: bool = False) -> ArosConfig:
    """Convenience accessor for AROS defaults."""
    return get_config_manager(force_reload=force_reload).get_aros_config()


def is_ci_environment() -> bool:
    """Check if we're running in a CI environment.

    This function centralizes CI detection by checking the CI environment variable,
    which is set automatically by CI systems (GitHub Actions, GitLab CI, etc.).

    The CI flag is external infrastructure configuration (not user-configurable via
    flujo.toml), but this helper provides access through the config_manager module
    to maintain a single entry point for configuration-related queries.

    Returns:
        True if running in a CI environment (CI env var set), False otherwise.
    """
    return os.environ.get("CI", "").lower() in ("true", "1", "yes")


# Wire domain-level config provider interface to avoid direct infra imports in domain logic
set_default_config_provider_fn: Optional[Callable[[Any], None]]
try:  # pragma: no cover - import guard
    from flujo.domain.interfaces import (
        set_default_config_provider as _set_default_config_provider_fn,
    )

    set_default_config_provider_fn = _set_default_config_provider_fn
except Exception:  # pragma: no cover - defensive fallback
    set_default_config_provider_fn = None


class _ConfigProviderAdapter:
    def load_config(self) -> Any:  # pragma: no cover - simple delegation
        return ConfigManager().load_config()


if set_default_config_provider_fn is not None:  # pragma: no cover - simple wiring
    try:
        set_default_config_provider_fn(_ConfigProviderAdapter())
    except Exception:
        pass
