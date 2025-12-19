"""Integration tests for the configuration management system."""

import pytest
import tempfile
import os
from pathlib import Path

from flujo.infra.config_manager import ConfigManager, ConfigurationError


class TestConfigurationIntegration:
    """Test the configuration system integration with CLI."""

    def test_configuration_file_loading_integration(self):
        """Test that configuration files are properly loaded and applied."""
        config_content = """
        [settings]
        max_iters = 10
        default_solution_model = "anthropic:claude-3-sonnet"
        reflection_enabled = false

        [solve]
        max_iters = 5
        k = 2
        reflection = true
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            # The settings should be overridden by the config file
            # Note: In a real scenario, the config manager would be initialized with the file
            # For this test, we'll verify the config manager can load the file
            config_manager = ConfigManager(config_path)
            config = config_manager.load_config()

            assert config.settings is not None
            assert config.settings.max_iters == 10
            assert config.settings.default_solution_model == "anthropic:claude-3-sonnet"
            assert config.settings.reflection_enabled is False

            assert config.solve is not None
            assert config.solve.max_iters == 5
            assert config.solve.k == 2
            assert config.solve.reflection is True
        finally:
            os.unlink(config_path)

    def test_cli_defaults_integration(self):
        """Test that CLI defaults are properly loaded from configuration."""
        config_content = """
        [solve]
        max_iters = 7
        k = 3
        solution_model = "openai:gpt-4o"

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
            assert solve_defaults["max_iters"] == 7
            assert solve_defaults["k"] == 3
            assert solve_defaults["solution_model"] == "openai:gpt-4o"

            # Test bench defaults
            bench_defaults = config_manager.get_cli_defaults("bench")
            assert bench_defaults["rounds"] == 15

            # Test run defaults
            run_defaults = config_manager.get_cli_defaults("run")
            assert run_defaults["pipeline_name"] == "my_pipeline"
            assert run_defaults["json_output"] is True
        finally:
            os.unlink(config_path)

    def test_configuration_precedence(self):
        """Test that configuration precedence works correctly."""
        config_content = """
        [settings]
        max_iters = 5
        default_solution_model = "openai:gpt-4o"

        [solve]
        max_iters = 3  # This should override the settings value
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config_manager = ConfigManager(config_path)
            config = config_manager.load_config()

            # The solve section should override the settings section for solve-specific values
            assert config.settings.max_iters == 5  # Global setting
            assert config.solve.max_iters == 3  # Command-specific override
        finally:
            os.unlink(config_path)

    def test_state_uri_configuration(self, monkeypatch: pytest.MonkeyPatch):
        """Test that state URI is properly configured."""
        config_content = """
        state_uri = "sqlite:///custom_flujo.db"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            monkeypatch.delenv("FLUJO_STATE_URI", raising=False)
            config_manager = ConfigManager(config_path)
            state_uri = config_manager.get_state_uri()

            assert state_uri == "sqlite:///custom_flujo.db"
        finally:
            os.unlink(config_path)

    def test_configuration_file_discovery(self):
        """Test that configuration files are discovered automatically."""
        config_content = """
        [settings]
        max_iters = 8
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
                assert config.settings.max_iters == 8
            finally:
                os.chdir(original_cwd)

    def test_backward_compatibility(self):
        """Test that the system maintains backward compatibility."""
        # Test that the system works without a configuration file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory to avoid finding flujo.toml in parent directories
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                config_manager = ConfigManager()
                config = config_manager.load_config()

                # Should return empty config when no file is found
                assert config.solve is None
                assert config.bench is None
                assert config.run is None
                assert config.settings is None
                assert config.state_uri is None
            finally:
                os.chdir(original_cwd)

    def test_error_handling(self):
        """Test that configuration errors are handled gracefully."""
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
            with pytest.raises(ConfigurationError):
                config_manager.load_config()
        finally:
            os.unlink(config_path)

    def test_missing_config_file_handling(self):
        """Test that missing configuration files are handled properly."""
        with pytest.raises(ConfigurationError):
            ConfigManager("nonexistent.toml")
