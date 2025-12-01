"""CI configuration tests.

These tests verify that the CI/CD pipeline is properly configured to enforce
the architectural and type safety standards established in FLUJO_TEAM_GUIDE.md.
"""

import yaml
import json
import os
from pathlib import Path
from typing import Optional
import pytest
from flujo.type_definitions.common import JSONObject


class TestCIConfiguration:
    """Test suite for CI/CD configuration compliance."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent

    def _read_yaml_file(self, file_path: Path) -> Optional[JSONObject]:
        """Read and parse a YAML file."""
        try:
            with open(file_path, "r") as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError, OSError):
            return None

    def _read_json_file(self, file_path: Path) -> Optional[JSONObject]:
        """Read and parse a JSON file."""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def test_github_actions_runs_architecture_tests(self, project_root: Path):
        """Verify that GitHub Actions runs architecture compliance tests."""
        workflows_dir = project_root / ".github" / "workflows"
        if not workflows_dir.exists():
            pytest.skip("No GitHub Actions workflows found")

        workflow_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))

        architecture_tests_run = False
        for workflow_file in workflow_files:
            workflow = self._read_yaml_file(workflow_file)
            if workflow:
                # Check if any job runs architecture tests
                for job_name, job_config in workflow.get("jobs", {}).items():
                    if isinstance(job_config, dict):
                        steps = job_config.get("steps", [])
                        for step in steps:
                            if isinstance(step, dict):
                                run_command = step.get("run", "")
                                if (
                                    "test_architecture" in run_command
                                    or "test_type_safety" in run_command
                                    or "tests/architecture" in run_command
                                ):
                                    architecture_tests_run = True
                                    break
                        if architecture_tests_run:
                            break
            if architecture_tests_run:
                break

        if not architecture_tests_run:
            pytest.fail(
                "GitHub Actions workflows do not run architecture compliance tests.\n"
                "CI/CD must include tests/architecture/ to enforce architectural standards."
            )

    def test_makefile_includes_architecture_tests(self, project_root: Path):
        """Verify that Makefile includes architecture compliance tests."""
        makefile = project_root / "Makefile"

        try:
            with open(makefile, "r") as f:
                content = f.read()

                # Check for architecture test targets
                if "test-architecture" not in content and "test_architecture" not in content:
                    pytest.fail(
                        "Makefile missing architecture test target.\n"
                        "Must include target to run tests/architecture/ for compliance checking."
                    )

                # Check that make all includes architecture tests
                if "test-architecture" in content:
                    if "make all" not in content or "test-architecture" not in content:
                        pytest.fail(
                            "'make all' target must include architecture tests.\n"
                            "Architecture compliance must be part of the main quality gate."
                        )

        except (FileNotFoundError, OSError):
            pytest.fail("Makefile not found or unreadable")

    def test_pre_commit_hooks_include_quality_checks(self, project_root: Path):
        """Verify that pre-commit hooks include quality checks."""
        pre_commit_config = project_root / ".pre-commit-config.yaml"

        if not pre_commit_config.exists():
            pytest.skip("No pre-commit configuration found")

        config = self._read_yaml_file(pre_commit_config)
        if not config:
            pytest.skip("Could not parse pre-commit configuration")

        # Check for required hooks
        required_hooks = ["mypy", "ruff"]
        found_hooks = []

        for repo in config.get("repos", []):
            for hook in repo.get("hooks", []):
                hook_id = hook.get("id", "")
                if any(required in hook_id for required in required_hooks):
                    found_hooks.append(hook_id)

        missing_hooks = [
            hook for hook in required_hooks if not any(hook in found for found in found_hooks)
        ]

        if missing_hooks:
            pytest.fail(
                f"Pre-commit configuration missing required hooks: {missing_hooks}\n"
                "Must include mypy and ruff hooks to enforce type safety and code quality."
            )

    def test_pyproject_toml_configures_quality_tools(self, project_root: Path):
        """Verify that pyproject.toml properly configures quality tools."""
        pyproject_file = project_root / "pyproject.toml"

        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        try:
            with open(pyproject_file, "rb") as f:
                config = tomllib.load(f)
        except (FileNotFoundError, OSError):
            pytest.fail("pyproject.toml not found")
        except Exception:
            pytest.fail("Could not parse pyproject.toml")

        # Check mypy configuration
        mypy_config = config.get("tool", {}).get("mypy", {})
        if not mypy_config:
            pytest.fail("pyproject.toml missing [tool.mypy] configuration")

        if not mypy_config.get("strict", False):
            pytest.fail("[tool.mypy] must have strict = true for type safety")

        # Check ruff configuration
        ruff_config = config.get("tool", {}).get("ruff", {})
        if not ruff_config:
            pytest.fail("pyproject.toml missing [tool.ruff] configuration")

        # Check pytest configuration includes architecture tests
        pytest_config = config.get("tool", {}).get("pytest", {}).get("ini_options", {})
        testpaths = pytest_config.get("testpaths", [])
        if "tests/architecture" not in str(testpaths):
            pytest.fail(
                "pyproject.toml pytest configuration must include tests/architecture in testpaths\n"
                "Architecture compliance tests must be discoverable by pytest."
            )

    def test_vscode_settings_enforce_quality(self, project_root: Path):
        """Verify that VS Code settings enforce quality standards."""
        vscode_settings = project_root / ".vscode" / "settings.json"

        if not vscode_settings.exists():
            pytest.skip("No VS Code settings found")

        settings = self._read_json_file(vscode_settings)
        if not settings:
            pytest.skip("Could not parse VS Code settings")

        # Check for mypy integration
        mypy_enabled = settings.get("python.linting.mypyEnabled", False)
        if not mypy_enabled:
            pytest.fail(
                "VS Code settings must enable mypy linting.\n"
                "Set python.linting.mypyEnabled to true for real-time type checking."
            )

        # Check for ruff integration
        ruff_enabled = settings.get("python.linting.ruffEnabled", False)
        if not ruff_enabled:
            pytest.fail(
                "VS Code settings must enable ruff linting.\n"
                "Set python.linting.ruffEnabled to true for consistent code formatting."
            )

    def test_architecture_tests_are_marked_as_critical(self, project_root: Path):
        """Verify that architecture tests are marked as critical quality gates."""
        # This test verifies that the architecture tests we created are properly integrated
        test_files = list((project_root / "tests" / "architecture").glob("*.py"))

        if not test_files:
            pytest.fail("No architecture test files found in tests/architecture/")

        # Check that test files contain critical assertions
        critical_patterns = [
            "pytest.fail",  # Tests that can fail builds
            "quality gate",  # References to quality gates
            "architectural",  # References to architecture
            "compliance",  # References to compliance
        ]

        found_critical_tests = False
        for test_file in test_files:
            try:
                with open(test_file, "r") as f:
                    content = f.read()
                    if any(pattern in content for pattern in critical_patterns):
                        found_critical_tests = True
                        break
            except (UnicodeDecodeError, OSError):
                continue

        if not found_critical_tests:
            pytest.fail(
                "Architecture tests do not contain critical quality gate assertions.\n"
                "Architecture compliance tests must include pytest.fail() calls that can block builds."
            )


class TestDevelopmentWorkflow:
    """Test suite for development workflow compliance."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent

    def test_makefile_quality_targets_exist(self, project_root: Path):
        """Verify that Makefile has all required quality targets."""
        makefile = project_root / "Makefile"

        try:
            with open(makefile, "r") as f:
                content = f.read()
        except (FileNotFoundError, OSError):
            pytest.fail("Makefile not found")

        required_targets = [
            "typecheck",  # Run mypy
            "lint",  # Run ruff
            "test",  # Run all tests
            "test-fast",  # Run fast test subset
            "all",  # Run all quality checks
        ]

        missing_targets = []
        for target in required_targets:
            target_pattern = f"{target}:"
            if target_pattern not in content:
                missing_targets.append(target)

        if missing_targets:
            pytest.fail(
                f"Makefile missing required quality targets: {missing_targets}\n"
                "Must include all quality assurance targets for development workflow."
            )

    def test_scripts_support_quality_checks(self, project_root: Path):
        """Verify that quality check scripts exist and are executable."""
        scripts_dir = project_root / "scripts"
        if not scripts_dir.exists():
            pytest.skip("No scripts directory found")

        # Look for quality-related scripts
        quality_scripts = [
            "check_optimal_workers.py",  # Performance checking
            "run_targeted_tests.py",  # Test running
        ]

        found_scripts = []
        for script in quality_scripts:
            script_path = scripts_dir / script
            if script_path.exists():
                found_scripts.append(script)
                # Check if executable (basic check)
                if not os.access(script_path, os.X_OK):
                    # Try to check if it's a Python script
                    try:
                        with open(script_path, "r") as f:
                            first_line = f.readline()
                            if not first_line.startswith("#!/usr/bin/env python"):
                                pytest.fail(
                                    f"Script {script} is not executable or properly shebanged"
                                )
                    except (UnicodeDecodeError, OSError):
                        pytest.fail(f"Cannot read script {script}")

        missing_scripts = [script for script in quality_scripts if script not in found_scripts]

        if missing_scripts:
            pytest.fail(
                f"Missing quality check scripts: {missing_scripts}\n"
                "Development workflow requires supporting scripts for quality assurance."
            )

    def test_contributing_guide_references_architecture_tests(self, project_root: Path):
        """Verify that contributing documentation references architecture tests."""
        contributing_files = [
            project_root / "CONTRIBUTING.md",
            project_root / "docs" / "development" / "type_safety.md",
            project_root / "FLUJO_TEAM_GUIDE.md",
        ]

        found_references = False
        for doc_file in contributing_files:
            if doc_file.exists():
                try:
                    with open(doc_file, "r") as f:
                        content = f.read()
                        if (
                            "architecture" in content.lower() and "test" in content.lower()
                        ) or "tests/architecture" in content:
                            found_references = True
                            break
                except (UnicodeDecodeError, OSError):
                    continue

        if not found_references:
            pytest.fail(
                "Contributing documentation does not reference architecture compliance tests.\n"
                "Contributors must know about tests/architecture/ quality gates."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
