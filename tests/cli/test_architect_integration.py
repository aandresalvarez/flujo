"""Integration tests for the Flujo Architect CLI that focus on user value.

These tests verify that the Architect CLI actually works for users:
- CLI commands work as documented
- Help system works correctly
- Error handling is graceful
- Basic functionality is accessible
"""

import pytest
from typer.testing import CliRunner
from flujo.cli.main import app

# Mark all tests in this module as slow (architect CLI integration tests)
pytestmark = [pytest.mark.slow]


class TestArchitectCLIIntegration:
    """Test the Architect CLI end-to-end functionality."""

    def test_architect_cli_help(self) -> None:
        """Test that CLI help commands work correctly."""
        runner = CliRunner()

        # Test main help
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0, "Main help should work"
        assert "create" in result.output, "Help should mention create command"

        # Test create command help
        result = runner.invoke(app, ["create", "--help"])
        assert result.exit_code == 0, "Create help should work"
        assert "goal" in result.output, "Create help should mention goal parameter"

    def test_architect_cli_error_handling(self) -> None:
        """Test that CLI handles various error conditions gracefully."""
        runner = CliRunner()

        # Test missing required arguments
        result = runner.invoke(app, ["create"])
        assert result.exit_code != 0, "CLI should fail without required arguments"

        # Test invalid command
        result = runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0, "CLI should fail with invalid command"

        # Test help for invalid command
        result = runner.invoke(app, ["invalid-command", "--help"])
        assert result.exit_code != 0, "Invalid command help should fail"

    def test_architect_cli_parameter_validation(self) -> None:
        """Test that CLI properly validates parameters."""
        runner = CliRunner()

        # Test with empty goal
        result = runner.invoke(app, ["create", "--goal", "", "--non-interactive"])
        assert result.exit_code != 0, "CLI should fail with empty goal"

        # Test with missing output directory
        result = runner.invoke(app, ["create", "--goal", "demo", "--non-interactive"])
        assert result.exit_code != 0, "CLI should fail without output directory"

    def test_architect_cli_name_parameter_help(self) -> None:
        """Test that CLI help includes information about name parameter."""
        runner = CliRunner()

        # Test create command help
        result = runner.invoke(app, ["create", "--help"])
        assert result.exit_code == 0, "Create help should work"

        # Help should mention name parameter
        help_text = result.output.lower()
        assert "name" in help_text, "Help should mention name parameter"

    def test_architect_cli_goal_parameter_help(self) -> None:
        """Test that CLI help includes information about goal parameter."""
        runner = CliRunner()

        # Test create command help
        result = runner.invoke(app, ["create", "--help"])
        assert result.exit_code == 0, "Create help should work"

        # Help should mention goal parameter
        help_text = result.output.lower()
        assert "goal" in help_text, "Help should mention goal parameter"

    def test_architect_cli_output_directory_help(self) -> None:
        """Test that CLI help includes information about output directory."""
        runner = CliRunner()

        # Test create command help
        result = runner.invoke(app, ["create", "--help"])
        assert result.exit_code == 0, "Create help should work"

        # Help should mention output directory
        help_text = result.output.lower()
        assert "output-dir" in help_text, "Help should mention output directory"

    def test_architect_cli_non_interactive_help(self) -> None:
        """Test that CLI help includes information about non-interactive mode."""
        runner = CliRunner()

        # Test create command help
        result = runner.invoke(app, ["create", "--help"])
        assert result.exit_code == 0, "Create help should work"

        # Help should mention non-interactive mode
        help_text = result.output.lower()
        assert "non-interactive" in help_text, "Help should mention non-interactive mode"

    def test_architect_cli_allow_side_effects_help(self) -> None:
        """Test that CLI help includes information about allow-side-effects flag."""
        runner = CliRunner()

        # Test create command help
        result = runner.invoke(app, ["create", "--help"])
        assert result.exit_code == 0, "Create help should work"

        # Help should mention allow-side-effects flag
        help_text = result.output.lower()
        assert "allow-side-effects" in help_text, "Help should mention allow-side-effects flag"

    def test_architect_cli_budget_parameter_help(self) -> None:
        """Test that CLI help includes information about budget parameter."""
        runner = CliRunner()

        # Test create command help
        result = runner.invoke(app, ["create", "--help"])
        assert result.exit_code == 0, "Create help should work"

        # Help should mention budget parameter
        help_text = result.output.lower()
        assert "budget" in help_text, "Help should mention budget parameter"

    def test_architect_cli_version_help(self) -> None:
        """Test that CLI help includes information about version."""
        runner = CliRunner()

        # Test main help
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0, "Main help should work"

        # Help should mention version
        help_text = result.output.lower()
        assert "version" in help_text, "Help should mention version"

    def test_architect_cli_command_structure(self) -> None:
        """Test that CLI has the expected command structure."""
        runner = CliRunner()

        # Test main help
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0, "Main help should work"

        # Should have create command
        help_text = result.output.lower()
        assert "create" in help_text, "CLI should have create command"

        # Should mention it's for creating pipelines
        assert "pipeline" in help_text, "CLI should mention pipeline creation"
