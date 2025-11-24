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
        """Top-level and create help should mention the core flags once."""
        runner = CliRunner()

        # Test main help
        main_help = runner.invoke(app, ["--help"])
        assert main_help.exit_code == 0, "Main help should work"
        main_text = main_help.output.lower()
        assert "create" in main_text, "Help should mention create command"
        assert "version" in main_text, "Help should mention version command"

        # Test create command help once and verify all key parameters appear
        create_help = runner.invoke(app, ["create", "--help"])
        assert create_help.exit_code == 0, "Create help should work"
        help_text = create_help.output.lower()
        for keyword in [
            "goal",
            "name",
            "output-dir",
            "non-interactive",
            "allow-side-effects",
            "budget",
        ]:
            assert keyword in help_text, f"Create help should mention '{keyword}'"

    def test_architect_cli_error_handling(self) -> None:
        """CLI should surface errors for invalid or missing parameters."""
        runner = CliRunner()

        # Missing required arguments should fail
        result = runner.invoke(app, ["create"])
        assert result.exit_code != 0, "CLI should fail without required arguments"

        # Empty goal should fail
        empty_goal = runner.invoke(app, ["create", "--goal", "", "--non-interactive"])
        assert empty_goal.exit_code != 0, "CLI should fail with empty goal"

        # Missing output directory should fail
        missing_output = runner.invoke(app, ["create", "--goal", "demo", "--non-interactive"])
        assert missing_output.exit_code != 0, "CLI should require output directory"

        # Invalid command should surface error
        invalid = runner.invoke(app, ["invalid-command"])
        assert invalid.exit_code != 0, "CLI should fail with invalid command"
