"""Integration tests for Flujo Architect CLI that focus on user value.

These tests verify that:
- The CLI works correctly with different goals
- Help system provides useful information
- Error handling is graceful
- Basic functionality is accessible
"""

from typer.testing import CliRunner
from flujo.cli.main import app


class TestArchitectCLIIntegration:
    """Test Architect CLI integration functionality."""

    def test_architect_cli_help_includes_expected_options(self) -> None:
        """Test that CLI help includes all expected options."""
        runner = CliRunner()

        # Test create command help
        result = runner.invoke(app, ["create", "--help"])
        assert result.exit_code == 0, "Create help should work"

        # Help should mention all expected parameters
        help_text = result.output.lower()
        expected_options = [
            "goal",
            "name",
            "budget",
            "output-dir",
            "non-interactive",
            "allow-side-effects",
        ]

        for option in expected_options:
            assert option in help_text, f"Help should mention {option} parameter"

    def test_architect_cli_goal_parameter_validation(self) -> None:
        """Test that the CLI properly validates goal parameters."""
        runner = CliRunner()

        # Test with empty goal
        result = runner.invoke(app, ["create", "--goal", "", "--non-interactive"])
        assert result.exit_code != 0, "CLI should fail with empty goal"

        # Test with missing goal
        result = runner.invoke(app, ["create", "--non-interactive"])
        assert result.exit_code != 0, "CLI should fail without goal"

    def test_architect_cli_output_directory_validation(self) -> None:
        """Test that the CLI properly validates output directory parameters."""
        runner = CliRunner()

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

    def test_architect_cli_budget_parameter_help(self) -> None:
        """Test that CLI help includes information about budget parameter."""
        runner = CliRunner()

        # Test create command help
        result = runner.invoke(app, ["create", "--help"])
        assert result.exit_code == 0, "Create help should work"

        # Help should mention budget parameter
        help_text = result.output.lower()
        assert "budget" in help_text, "Help should mention budget parameter"

    def test_architect_cli_context_file_parameter_help(self) -> None:
        """Test that CLI help includes information about context file parameter."""
        runner = CliRunner()

        # Test create command help
        result = runner.invoke(app, ["create", "--help"])
        assert result.exit_code == 0, "Create help should work"

        # Help should mention context file parameter
        help_text = result.output.lower()
        assert "context-file" in help_text, "Help should mention context file parameter"

    def test_architect_cli_force_parameter_help(self) -> None:
        """Test that CLI help includes information about force parameter."""
        runner = CliRunner()

        # Test create command help
        result = runner.invoke(app, ["create", "--help"])
        assert result.exit_code == 0, "Create help should work"

        # Help should mention force parameter
        help_text = result.output.lower()
        assert "force" in help_text, "Help should mention force parameter"

    def test_architect_cli_strict_parameter_help(self) -> None:
        """Test that CLI help includes information about strict parameter."""
        runner = CliRunner()

        # Test create command help
        result = runner.invoke(app, ["create", "--help"])
        assert result.exit_code == 0, "Create help should work"

        # Help should mention strict parameter
        help_text = result.output.lower()
        assert "strict" in help_text, "Help should mention strict parameter"

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

    def test_architect_cli_parameter_defaults(self) -> None:
        """Test that CLI shows parameter defaults in help."""
        runner = CliRunner()

        # Test create command help
        result = runner.invoke(app, ["create", "--help"])
        assert result.exit_code == 0, "Create help should work"

        # Help should show default values
        help_text_raw = result.output
        help_text = help_text_raw.lower()
        # Typer/Click versions may render None as "none" or "None"; accept both.
        assert (
            "[default: none]" in help_text
            or "[default: None]" in help_text_raw
            or "default:" in help_text
        ), f"Help should show default values; got: {help_text_raw!r}"

    def test_architect_cli_parameter_types(self) -> None:
        """Test that CLI shows parameter types in help."""
        runner = CliRunner()

        # Test create command help
        result = runner.invoke(app, ["create", "--help"])
        assert result.exit_code == 0, "Create help should work"

        # Help should show parameter types
        help_text = result.output.lower()
        assert "text" in help_text, "Help should show parameter types"
        assert "float" in help_text, "Help should show parameter types"
