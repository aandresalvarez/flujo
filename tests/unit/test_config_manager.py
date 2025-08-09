"""Tests for the configuration manager."""

import pytest
import tempfile
import os
from pathlib import Path
from flujo.infra.config_manager import (
    ConfigManager,
    FlujoConfig,
    SolveConfig,
    BenchConfig,
    SettingsOverrides,
)
from flujo.exceptions import ConfigurationError
import unittest.mock


class TestConfigManager:
    """Test the configuration manager functionality."""

    def test_empty_config(self):
        """Test that empty configuration works correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory to avoid finding flujo.toml in parent directories
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                config_manager = ConfigManager()
                config = config_manager.load_config()

                assert isinstance(config, FlujoConfig)
                assert config.solve is None
                assert config.bench is None
                assert config.run is None
                assert config.settings is None
                assert config.state_uri is None
            finally:
                os.chdir(original_cwd)

    def test_basic_config_loading(self):
        """Test loading a basic configuration file."""
        config_content = """
        state_uri = "sqlite:///test.db"

        [settings]
        max_iters = 10
        default_solution_model = "openai:gpt-4o"

        [solve]
        max_iters = 5
        k = 2
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config_manager = ConfigManager(config_path)
            config = config_manager.load_config()

            assert config.state_uri == "sqlite:///test.db"
            assert config.settings is not None
            assert config.settings.max_iters == 10
            assert config.settings.default_solution_model == "openai:gpt-4o"
            assert config.solve is not None
            assert config.solve.max_iters == 5
            assert config.solve.k == 2
        finally:
            os.unlink(config_path)

    def test_cli_defaults(self):
        """Test getting CLI defaults from configuration."""
        config_content = """
        [solve]
        max_iters = 5
        k = 2
        reflection = true

        [bench]
        rounds = 15

        [run]
        pipeline_name = "my_pipeline"
        json_output = true
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config_manager = ConfigManager(config_path)

            # Test solve defaults
            solve_defaults = config_manager.get_cli_defaults("solve")
            assert solve_defaults["max_iters"] == 5
            assert solve_defaults["k"] == 2
            assert solve_defaults["reflection"] is True

            # Test bench defaults
            bench_defaults = config_manager.get_cli_defaults("bench")
            assert bench_defaults["rounds"] == 15

            # Test run defaults
            run_defaults = config_manager.get_cli_defaults("run")
            assert run_defaults["pipeline_name"] == "my_pipeline"
            assert run_defaults["json_output"] is True

            # Test non-existent command
            empty_defaults = config_manager.get_cli_defaults("nonexistent")
            assert empty_defaults == {}
        finally:
            os.unlink(config_path)

    def test_settings_override(self):
        """Test that settings overrides work correctly."""
        config_content = """
        [settings]
        max_iters = 10
        default_solution_model = "anthropic:claude-3-sonnet"
        reflection_enabled = false
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config_manager = ConfigManager(config_path)
            settings = config_manager.get_settings()

            assert settings.max_iters == 10
            assert settings.default_solution_model == "anthropic:claude-3-sonnet"
            assert settings.reflection_enabled is False
        finally:
            os.unlink(config_path)

    def test_config_file_discovery(self):
        """Test automatic configuration file discovery."""
        config_content = """
        [settings]
        max_iters = 5
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_file = temp_path / "flujo.toml"

            with open(config_file, "w") as f:
                f.write(config_content)

            # Change to the temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                config_manager = ConfigManager()
                config = config_manager.load_config()

                assert config.settings is not None
                assert config.settings.max_iters == 5
            finally:
                os.chdir(original_cwd)

    def test_parent_directory_search(self):
        """Test searching parent directories for configuration files."""
        config_content = """
        [settings]
        max_iters = 10
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            parent_dir = temp_path / "parent"
            child_dir = parent_dir / "child"

            # Create parent directory with config
            parent_dir.mkdir()
            config_file = parent_dir / "flujo.toml"
            with open(config_file, "w") as f:
                f.write(config_content)

            # Create child directory
            child_dir.mkdir()

            # Change to child directory
            original_cwd = os.getcwd()
            try:
                os.chdir(child_dir)

                config_manager = ConfigManager()
                config = config_manager.load_config()

                assert config.settings is not None
                assert config.settings.max_iters == 10
            finally:
                os.chdir(original_cwd)

    def test_invalid_config_file(self):
        """Test handling of invalid configuration files."""
        # Invalid type for max_iters to trigger validation error
        invalid_content = """
        [settings]
        max_iters = "not_a_number"  # Invalid type
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(invalid_content)
            config_path = f.name

        try:
            config_manager = ConfigManager(config_path)
            # Should raise an exception due to invalid type
            with pytest.raises(
                ConfigurationError,
                match="Error loading configuration",
            ):
                config_manager.load_config()
        finally:
            os.unlink(config_path)

    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            ConfigManager("nonexistent.toml")

    def test_global_functions(self):
        """Test the global configuration functions."""
        config_content = """
        state_uri = "sqlite:///test.db"

        [solve]
        max_iters = 5
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        # Isolate environment variables for this test
        original_state_uri = os.environ.get("FLUJO_STATE_URI")
        try:
            # Remove any existing FLUJO_STATE_URI to test TOML file behavior
            os.environ.pop("FLUJO_STATE_URI", None)
            
            # Test with a real config manager using our test file
            config_manager = ConfigManager(config_path)

            # Test CLI defaults
            defaults = config_manager.get_cli_defaults("solve")
            assert defaults["max_iters"] == 5

            # Test state URI
            uri = config_manager.get_state_uri()
            assert uri == "sqlite:///test.db"
        finally:
            # Restore original environment
            if original_state_uri is not None:
                os.environ["FLUJO_STATE_URI"] = original_state_uri
            os.unlink(config_path)

    def test_state_uri_environment_precedence(self):
        """Test that environment variables take precedence over TOML file for state_uri."""
        config_content = """
        state_uri = "sqlite:///test.db"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        original_state_uri = os.environ.get("FLUJO_STATE_URI")
        try:
            # Set environment variable to override TOML value
            os.environ["FLUJO_STATE_URI"] = "sqlite:///env.db"
            
            config_manager = ConfigManager(config_path)
            
            # Environment variable should take precedence
            uri = config_manager.get_state_uri()
            assert uri == "sqlite:///env.db"
            
            # Remove environment variable to test TOML fallback
            os.environ.pop("FLUJO_STATE_URI", None)
            
            # Now should get TOML value
            uri = config_manager.get_state_uri(force_reload=True)
            assert uri == "sqlite:///test.db"
            
        finally:
            # Restore original environment
            if original_state_uri is not None:
                os.environ["FLUJO_STATE_URI"] = original_state_uri
            else:
                os.environ.pop("FLUJO_STATE_URI", None)
            os.unlink(config_path)

    def test_settings_integration(self):
        """Test integration with the settings system."""
        config_content = """
        [settings]
        max_iters = 15
        default_solution_model = "anthropic:claude-3-sonnet"
        reflection_enabled = false
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            with unittest.mock.patch.dict(os.environ, {"K_VARIANTS": "3"}):
                config_manager = ConfigManager(config_path)
                settings = config_manager.get_settings()

                # Verify settings are overridden
                assert settings.max_iters == 15
                assert settings.default_solution_model == "anthropic:claude-3-sonnet"
                assert settings.reflection_enabled is False

                # Verify other settings remain at defaults
                assert settings.k_variants == 3
                assert settings.agent_timeout == 60  # Default value
        finally:
            os.unlink(config_path)

    def test_invalid_k_variants_env(self):
        """Test that invalid K_VARIANTS env var raises or falls back to default."""
        config_content = """
        [settings]
        max_iters = 15
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = f.name
        try:
            with unittest.mock.patch.dict(os.environ, {"K_VARIANTS": "not_an_int"}):
                config_manager = ConfigManager(config_path)
                with pytest.raises((ValueError, TypeError)):
                    config_manager.get_settings()
        finally:
            os.unlink(config_path)


class TestConfigurationModels:
    """Test the configuration data models."""

    def test_flujo_config(self):
        """Test FlujoConfig model."""
        config = FlujoConfig(
            solve=SolveConfig(max_iters=5),
            bench=BenchConfig(rounds=10),
            settings=SettingsOverrides(max_iters=15),
            state_uri="sqlite:///test.db",
        )

        assert config.solve.max_iters == 5
        assert config.bench.rounds == 10
        assert config.settings.max_iters == 15
        assert config.state_uri == "sqlite:///test.db"

    def test_solve_config(self):
        """Test SolveConfig model."""
        solve_config = SolveConfig(
            max_iters=5,
            k=2,
            reflection=True,
            scorer="ratio",
            solution_model="openai:gpt-4o",
        )

        assert solve_config.max_iters == 5
        assert solve_config.k == 2
        assert solve_config.reflection is True
        assert solve_config.scorer == "ratio"
        assert solve_config.solution_model == "openai:gpt-4o"

    def test_settings_overrides(self):
        """Test SettingsOverrides model."""
        overrides = SettingsOverrides(
            max_iters=10,
            default_solution_model="anthropic:claude-3-sonnet",
            reflection_enabled=False,
        )

        assert overrides.max_iters == 10
        assert overrides.default_solution_model == "anthropic:claude-3-sonnet"
        assert overrides.reflection_enabled is False
