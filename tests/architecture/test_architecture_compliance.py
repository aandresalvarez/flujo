"""Architecture compliance tests.

These tests verify that the codebase follows the architectural patterns
established in FLUJO_TEAM_GUIDE.md, including policy-driven architecture,
exception handling patterns, and code quality standards.
"""

import os
import subprocess
import ast
import sys
from pathlib import Path
from typing import List, Optional
import pytest

# Mark the entire module as slow and allow ample time for the heavy quality gates.
# The subprocess stack (mypy + ruff + unit tests) can exceed 5 minutes under CI load,
# so we give a generous window to avoid false timeouts.
pytestmark = [pytest.mark.slow, pytest.mark.timeout(900)]


class TestArchitectureCompliance:
    """Test suite for architectural pattern compliance."""

    @pytest.fixture
    def flujo_root(self) -> Path:
        """Get the root directory of the Flujo project."""
        return Path(__file__).parent.parent.parent

    def _get_python_files(self, flujo_root: Path, package: str = "flujo") -> List[Path]:
        """Get all Python files in the specified package."""
        package_dir = flujo_root / package
        return list(package_dir.rglob("*.py"))

    def _parse_python_file(self, file_path: Path) -> Optional[ast.Module]:
        """Parse a Python file into an AST."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return ast.parse(content, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError, OSError):
            return None

    def _parse_python_file(self, file_path: Path) -> Optional[ast.Module]:
        """Parse a Python file into an AST."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return ast.parse(content, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError, OSError):
            return None

    def _grep_files(self, files: List[Path], pattern: str) -> List[str]:
        """Search for pattern in files using grep."""
        results = []
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines, 1):
                        if pattern in line:
                            results.append(f"{file_path}:{i}:{line.strip()}")
            except (UnicodeDecodeError, OSError):
                continue
        return results

    def test_policy_driven_architecture_is_followed(self, flujo_root: Path):
        """Verify that the policy-driven architecture is followed.

        All step-specific logic should be in policies, not in ExecutorCore.
        """
        executor_file = flujo_root / "flujo/application/core/executor_core.py"

        try:
            with open(executor_file, "r") as f:
                content = f.read()

                # Look for step-specific logic in the dispatcher
                execute_method_start = content.find("def execute(")
                if execute_method_start == -1:
                    pytest.fail("Could not find execute method in ExecutorCore")

                # Find the end of the execute method
                next_method = content.find("\n    def ", execute_method_start + 1)
                if next_method == -1:
                    execute_content = content[execute_method_start:]
                else:
                    execute_content = content[execute_method_start:next_method]

                # Check for problematic patterns
                violations = []

                # Look for direct step logic instead of delegation
                if "if isinstance(step," in execute_content and "self." in execute_content:
                    # This is OK if it's just delegation to policies
                    # But flag if there's actual logic implementation
                    lines = execute_content.split("\n")
                    for i, line in enumerate(lines):
                        if "if isinstance(step," in line and i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if not ("self." in next_line and "execute(" in next_line):
                                violations.append(f"Direct step logic in executor: {line.strip()}")

                if violations:
                    pytest.fail(
                        "Found violations of policy-driven architecture in ExecutorCore:\n"
                        + "\n".join(violations)
                        + "\n\nAll step-specific logic must be in policy classes, not ExecutorCore."
                    )

        except (UnicodeDecodeError, OSError) as e:
            pytest.fail(f"Could not read executor_core.py: {e}")

    def test_exception_handling_patterns_are_followed(self, flujo_root: Path):
        """Verify that exception handling follows the established patterns.

        Control flow exceptions should be re-raised, data exceptions should become failures.
        """
        policy_files = list((flujo_root / "flujo/application/core").glob("step_policies.py"))
        if not policy_files:
            # Try the single file approach
            policy_files = [flujo_root / "flujo/application/core/step_policies.py"]

        violations = []
        for policy_file in policy_files:
            try:
                tree = self._parse_python_file(policy_file)
                if tree is None:
                    continue

                for node in ast.walk(tree):
                    if isinstance(node, ast.ExceptHandler):
                        # Check exception handler body
                        for stmt in node.body:
                            if isinstance(stmt, ast.Return):
                                # Check if it's returning a StepResult in an exception handler
                                # This might be OK, but let's flag suspicious patterns
                                if isinstance(node.exc, ast.Name):
                                    exc_name = node.exc.id
                                    if exc_name in [
                                        "PausedException",
                                        "PipelineAbortSignal",
                                        "InfiniteRedirectError",
                                    ]:
                                        violations.append(
                                            f"{policy_file}: Control flow exception {exc_name} "
                                            "should be re-raised, not converted to StepResult"
                                        )

            except Exception as e:
                violations.append(f"Could not analyze {policy_file}: {e}")

        if violations:
            pytest.fail(
                f"Found {len(violations)} violations of exception handling patterns:\n"
                + "\n".join(violations)
                + "\n\nControl flow exceptions must be re-raised, not converted to StepResult failures."
            )

    def test_context_isolation_patterns_are_used(self, flujo_root: Path):
        """Verify that context isolation is used in complex step policies.

        LoopStep and ParallelStep policies should use ContextManager.isolate().
        """
        policy_files = [flujo_root / "flujo/application/core/step_policies.py"]

        violations = []
        for policy_file in policy_files:
            try:
                with open(policy_file, "r") as f:
                    content = f.read()

                    # Check for complex step classes
                    class_definitions = []
                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        if line.startswith("class ") and (
                            "LoopStepExecutor" in line or "ParallelStepExecutor" in line
                        ):
                            class_definitions.append((i, line))

                    for class_start, class_line in class_definitions:
                        # Find the end of the class
                        class_end = len(lines)
                        for j in range(class_start + 1, len(lines)):
                            if lines[j].startswith("class ") or (j == len(lines) - 1):
                                class_end = j
                                break

                        class_content = "\n".join(lines[class_start:class_end])

                        # Check for ContextManager.isolate usage
                        if "ContextManager.isolate" not in class_content:
                            class_name = class_line.split()[1].split("(")[0]
                            violations.append(
                                f"{policy_file}: {class_name} missing ContextManager.isolate()"
                            )

            except (UnicodeDecodeError, OSError) as e:
                violations.append(f"Could not analyze {policy_file}: {e}")

        if violations:
            pytest.fail(
                f"Found {len(violations)} complex step policies missing context isolation:\n"
                + "\n".join(violations)
                + "\n\nComplex steps (LoopStep, ParallelStep) must use ContextManager.isolate() for idempotency."
            )

    def test_quota_patterns_are_followed(self, flujo_root: Path):
        """Verify that quota patterns (Reserve → Execute → Reconcile) are followed.

        Policies that consume resources should follow the quota pattern.
        """
        policy_files = [flujo_root / "flujo/application/core/step_policies.py"]

        violations = []
        for policy_file in policy_files:
            try:
                with open(policy_file, "r") as f:
                    content = f.read()

                    # Look for quota usage without proper pattern
                    if "CURRENT_QUOTA" in content:
                        # Check if the pattern is followed
                        if not ("reserve" in content.lower() and "reconcile" in content.lower()):
                            violations.append(
                                f"{policy_file}: Uses CURRENT_QUOTA but missing Reserve→Execute→Reconcile pattern"
                            )

            except (UnicodeDecodeError, OSError) as e:
                violations.append(f"Could not analyze {policy_file}: {e}")

        if violations:
            pytest.fail(
                f"Found {len(violations)} quota usage violations:\n"
                + "\n".join(violations)
                + "\n\nAll resource consumption must follow: Reserve → Execute → Reconcile pattern."
            )

    def test_fallback_logic_is_properly_implemented(self, flujo_root: Path):
        """Verify that fallback logic follows established patterns.

        Fallback chains should be detected and limited.
        """
        policy_files = [flujo_root / "flujo/application/core/step_policies.py"]
        fallback_handler = flujo_root / "flujo/application/core/runtime/fallback_handler.py"

        violations = []
        for policy_file in policy_files:
            try:
                with open(policy_file, "r") as f:
                    content = f.read()

                    # Check for fallback logic without chain detection
                    if "fallback" in content.lower():
                        if "check_for_loop" not in content:
                            violations.append(
                                f"{policy_file}: Fallback logic without infinite loop detection"
                            )

            except (UnicodeDecodeError, OSError) as e:
                violations.append(f"Could not analyze {policy_file}: {e}")

        # Check fallback handler exists and has loop detection
        if not fallback_handler.exists():
            violations.append("Missing fallback_handler.py with infinite loop detection")
        else:
            try:
                with open(fallback_handler, "r") as f:
                    content = f.read()
                    if "check_for_loop" not in content:
                        violations.append("fallback_handler.py missing check_for_loop method")
            except (UnicodeDecodeError, OSError):
                violations.append("Could not read fallback_handler.py")

        if violations:
            pytest.fail(
                f"Found {len(violations)} fallback implementation violations:\n"
                + "\n".join(violations)
                + "\n\nAll fallback logic must include infinite loop detection."
            )

    def test_step_complexity_detection_is_used(self, flujo_root: Path):
        """Verify that step complexity detection is used appropriately.

        Steps should declare themselves as complex when they need policy routing.
        """
        step_files = list((flujo_root / "flujo/domain/dsl").glob("*.py"))

        violations = []
        for step_file in step_files:
            try:
                with open(step_file, "r") as f:
                    content = f.read()

                    # Look for Step subclasses
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            # Check if it inherits from Step
                            for base in node.bases:
                                if isinstance(base, ast.Name) and base.id == "Step":
                                    # Check if it has is_complex property
                                    has_complex_property = False
                                    for item in node.body:
                                        if (
                                            isinstance(item, ast.FunctionDef)
                                            and item.name == "is_complex"
                                        ):
                                            has_complex_property = True
                                            break

                                    if not has_complex_property:
                                        violations.append(
                                            f"{step_file}: {node.name} missing is_complex property"
                                        )

            except (SyntaxError, UnicodeDecodeError, OSError):
                continue

        if violations:
            pytest.fail(
                f"Found {len(violations)} Step subclasses missing is_complex property:\n"
                + "\n".join(violations)
                + "\n\nAll Step subclasses must define is_complex property for proper routing."
            )


class TestCodeQualityStandards:
    """Test suite for code quality standards from FLUJO_TEAM_GUIDE.md."""

    @pytest.fixture
    def flujo_root(self) -> Path:
        """Get the root directory of the Flujo project."""
        return Path(__file__).parent.parent.parent

    def _get_python_files(self, flujo_root: Path, package: str = "flujo") -> List[Path]:
        """Get all Python files in the specified package."""
        package_dir = flujo_root / package
        return list(package_dir.rglob("*.py"))

    def _parse_python_file(self, file_path: Path) -> Optional[ast.Module]:
        """Parse a Python file into an AST."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return ast.parse(content, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError, OSError):
            return None

    def test_all_functions_have_type_annotations(self, flujo_root: Path):
        """Verify that all functions have complete type annotations.

        This is a mandatory requirement from the type safety guide.
        """
        python_files = self._get_python_files(flujo_root)
        violations = []

        for file_path in python_files:
            try:
                tree = self._parse_python_file(file_path)
                if tree is None:
                    continue

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check function signature
                        if not node.returns:  # Missing return type
                            violations.append(
                                f"{file_path}: {node.name}() missing return type annotation"
                            )

                        # Check parameter annotations
                        for arg in node.args.args:
                            if arg.annotation is None and arg.arg != "self":
                                violations.append(
                                    f"{file_path}: {node.name}() parameter '{arg.arg}' missing type annotation"
                                )

            except Exception as e:
                violations.append(f"Could not analyze {file_path}: {e}")

        # Allow some baseline violations for legacy code
        max_allowed_violations = 100  # Should be reduced over time

        if len(violations) > max_allowed_violations:
            pytest.fail(
                f"Found {len(violations)} functions/methods without complete type annotations, "
                f"exceeding baseline of {max_allowed_violations}.\n"
                f"Type annotations are mandatory. First 10 violations:\n"
                + "\n".join(violations[:10])
            )

    def test_no_direct_flujo_toml_access(self, flujo_root: Path):
        """Verify that flujo.toml is not accessed directly.

        All configuration must go through ConfigManager.
        """
        python_files = self._get_python_files(flujo_root)
        violations = []

        for file_path in python_files:
            try:
                with open(file_path, "r") as f:
                    content = f.read()

                    # Look for direct flujo.toml access
                    if "flujo.toml" in content:
                        violations.append(f"{file_path}: Direct access to flujo.toml")

                    # Look for direct environment variable access for config
                    if "os.environ.get" in content and (
                        "FLUJO_" in content or "flujo" in content.lower()
                    ):
                        violations.append(
                            f"{file_path}: Direct environment variable access for config"
                        )

            except (UnicodeDecodeError, OSError):
                continue

        max_allowed_violations = 50  # Baseline legacy access to be reduced over time

        if len(violations) > max_allowed_violations:
            pytest.fail(
                f"Found {len(violations)} violations of configuration access patterns, "
                f"exceeding baseline of {max_allowed_violations}:\n"
                + "\n".join(violations)
                + "\n\nAll configuration must go through ConfigManager, not direct file/env access."
            )

    def test_legacy_warnings_are_enabled_in_ci(self, flujo_root: Path):
        """Verify legacy shims have been removed now that Phase 6 cleanup is underway."""

        warning_file = flujo_root / "flujo/exceptions.py"

        try:
            content = warning_file.read_text()
        except (UnicodeDecodeError, OSError):
            pytest.fail("Could not check legacy warning system")

        if "OrchestratorError" in content or "FlujoFrameworkError" in content:
            pytest.fail("Deprecated exception shims must be removed from flujo.exceptions")
        if "DeprecationWarning" in content:
            pytest.fail("Legacy deprecation warnings should not remain in flujo.exceptions")
        assert "FlujoError" in content, (
            "exceptions module should define FlujoError as the base type"
        )

    def test_serialization_is_unified(self, flujo_root: Path):
        """Verify that all models use unified serialization.

        All Flujo models should use the centralized serialization system.
        """
        model_files = [
            flujo_root / "flujo/domain/base_model.py",
            flujo_root / "flujo/domain/models.py",
        ]

        violations = []
        for model_file in model_files:
            try:
                with open(model_file, "r") as f:
                    content = f.read()

                    # Check for unified serialization
                    uses_unified_serialization = (
                        "_serialize_for_json" in content
                        or "model_dump" in content
                        or "from .base_model import BaseModel" in content
                    )
                    if not uses_unified_serialization:
                        violations.append(f"{model_file}: Not using unified serialization")

            except (UnicodeDecodeError, OSError):
                violations.append(f"Could not check {model_file}")

        if violations:
            pytest.fail(
                f"Found {len(violations)} models not using unified serialization:\n"
                + "\n".join(violations)
                + "\n\nAll models must use flujo.utils.serialization for consistency."
            )

    def test_legacy_safe_serialize_removed(self, flujo_root: Path) -> None:
        """Ensure the legacy safe_serialize helper is not reintroduced."""

        hits: list[str] = []
        for path in flujo_root.rglob("*.py"):
            # Skip this test file to avoid self-matching on the test name/docstring.
            if path == Path(__file__):
                continue
            try:
                content = path.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            if "safe_serialize" in content:
                hits.append(str(path.relative_to(flujo_root)))

        if hits:
            pytest.fail("Legacy safe_serialize references found:\n" + "\n".join(sorted(hits)))


class TestQualityGates:
    """Test suite for quality gates that must pass before PR merge."""

    @pytest.fixture
    def flujo_root(self) -> Path:
        """Get the root directory of the Flujo project."""
        return Path(__file__).parent.parent.parent

    def test_all_quality_checks_pass(self, flujo_root: Path):
        """Run all quality checks that must pass before PR merge.

        This is the comprehensive quality gate test.
        """
        if os.getenv("GITHUB_ACTIONS") == "true":
            pytest.skip(
                "Quality gates are enforced by the PR workflow jobs (Quality Checks / Fast Tests / Unit Tests)."
            )
        failures = []
        # Ensure quality gate subprocesses run with plugin autoload enabled so required plugins load.
        env = dict(os.environ)
        env.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)

        # Run mypy type checking
        try:
            print("Running mypy --strict flujo/ ...", flush=True)
            result = subprocess.run(
                [sys.executable, "-m", "mypy", "--strict", "flujo/"],
                cwd=flujo_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                failures.append(f"mypy failed: {result.stderr[:500]}...")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            failures.append("mypy check timed out or mypy not available")

        # Run linting
        try:
            print("Running ruff check flujo/ ...", flush=True)
            result = subprocess.run(
                [sys.executable, "-m", "ruff", "check", "flujo/"],
                cwd=flujo_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                failures.append(f"ruff check failed: {result.stderr[:500]}...")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            failures.append("ruff check timed out or ruff not available")

        # Run tests (fast subset, excluding serial tests that need isolation)
        try:
            print("Running unit tests (tests/unit/) ...", flush=True)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-x",
                    "--tb=short",
                    "-m",
                    "not slow and not veryslow and not serial and not benchmark",
                    "tests/unit/",
                ],
                cwd=flujo_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                error_info = f"unit tests failed:\nSTDOUT:\n{result.stdout[-1000:]}\nSTDERR:\n{result.stderr[-2000:]}"
                failures.append(error_info)
        except subprocess.TimeoutExpired:
            failures.append("unit tests timed out")

        if failures:
            pytest.fail(
                f"Quality gates failed with {len(failures)} issues:\n"
                + "\n".join(f"- {failure}" for failure in failures)
                + "\n\nAll quality gates must pass before PR merge."
            )

    def test_srp_compliance_basic_metrics(self, flujo_root: Path):
        """Verify basic SRP compliance through size and complexity metrics."""
        # Import SRP test classes
        try:
            from test_srp_compliance import TestSRPCompliance
            from test_srp_semantic_analysis import TestSRPSemanticCompliance
        except ImportError:
            # Fallback for when running as part of architecture tests
            import sys
            import os

            sys.path.insert(0, os.path.dirname(__file__))
            from test_srp_compliance import TestSRPCompliance
            from test_srp_semantic_analysis import TestSRPSemanticCompliance

        # Run SRP tests as part of architecture compliance
        srp_tester = TestSRPCompliance()
        semantic_tester = TestSRPSemanticCompliance()

        try:
            srp_tester.test_classes_follow_srp_method_count_limits(flujo_root)
            srp_tester.test_no_god_classes_by_method_count(flujo_root)
            semantic_tester.test_classes_have_focused_responsibilities(flujo_root)
            print("✓ SRP compliance checks passed")
        except Exception as e:
            pytest.fail(f"SRP compliance failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
