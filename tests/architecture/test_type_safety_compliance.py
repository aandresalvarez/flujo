"""Architecture tests to ensure type safety compliance.

These tests verify that the codebase maintains strong type safety and follows
the patterns established in docs/development/type_safety.md and FLUJO_TEAM_GUIDE.md.
"""

import os
import subprocess
from pathlib import Path
from typing import List
import pytest


MAKE_ALL_TIMEOUT_SECONDS = 900
FAILURE_SUMMARY_GLOB = "failure_summary_*.txt"


class TestTypeSafetyCompliance:
    """Test suite for type safety compliance."""

    @pytest.fixture
    def flujo_root(self) -> Path:
        """Get the root directory of the Flujo project."""
        return Path(__file__).parent.parent.parent

    def _get_python_files(self, root: Path) -> List[Path]:
        """Get all Python files in the flujo package."""
        flujo_dir = root / "flujo"
        return list(flujo_dir.rglob("*.py"))

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
                # Skip binary files or files we can't read
                continue
        return results

    def test_no_new_any_types_in_flujo_package(self, flujo_root: Path):
        """Verify that `Any` types are not introduced in new code.

        This test ensures that the type safety improvements are maintained
        and no new `Any` types are introduced in the flujo package.
        """
        python_files = self._get_python_files(flujo_root)
        any_occurrences = self._grep_files(python_files, "Any")

        # Filter out acceptable uses of Any
        acceptable_any_uses = {
            "from typing import Any",  # Import statements are OK
            "from typing import Any as",  # Import aliases are OK
            "import typing",  # typing module imports are OK
            "typing.Any",  # Explicit typing.Any references are OK
            "Any,",  # In import lists
            "Any as",  # Import aliases
            ": Any",  # Type annotations (these should be evaluated case-by-case)
            "-> Any",  # Return type annotations (should be evaluated)
            "[Any]",  # Generic types
            "Any]",  # End of generic types
            "Any)",  # End of function signatures
            "Any,",  # In tuples/lists
            "Any |",  # Union types
            "| Any",  # Union types
        }

        # Known acceptable files and locations (baseline)
        acceptable_locations = {
            # Type definition files
            "flujo/type_definitions/common.py",
            "flujo/type_definitions/validation.py",
            "flujo/type_definitions/__init__.py",
            # Test files for type safety
            "tests/test_types/",
            # Domain types (some legacy Any usage)
            "flujo/domain/types.py",
            "flujo/domain/models.py",
            "flujo/domain/dsl/step.py",
        }

        concerning_any_uses = []

        for occurrence in any_occurrences:
            file_path, line_num, line_content = occurrence.split(":", 2)
            file_path = file_path.strip()

            # Skip acceptable import statements
            if any(acceptable in line_content for acceptable in acceptable_any_uses):
                continue

            # Skip known acceptable files
            if any(acceptable in file_path for acceptable in acceptable_locations):
                continue

            # Flag concerning uses
            concerning_any_uses.append(occurrence)

        # Allow some baseline Any usage but prevent significant new usage
        # Lowered from 600 to 500 - current DSL Any ~443, prevent regression
        max_allowed_any = 500

        if len(concerning_any_uses) > max_allowed_any:
            pytest.fail(
                f"Found {len(concerning_any_uses)} concerning Any type usages, "
                f"exceeding the allowed baseline of {max_allowed_any}.\n"
                f"Type safety regression detected. New Any types introduced:\n"
                + "\n".join(concerning_any_uses[:10])  # Show first 10
                + (
                    f"\n... and {len(concerning_any_uses) - 10} more"
                    if len(concerning_any_uses) > 10
                    else ""
                )
            )

    def test_uses_jsonobject_instead_of_raw_dict_any(self, flujo_root: Path):
        """Verify that JSONObject is used instead of raw Dict[str, Any] in new code.

        This test ensures that the type safety improvement of using JSONObject
        aliases is followed in new code.
        """
        python_files = self._get_python_files(flujo_root)

        # Patterns that should be replaced by JSONObject or specific TypedDict/BaseModel
        dict_any_patterns = [
            "Dict[str, Any]",
            "dict[str, Any]",
            "Mapping[str, Any]",
            "MutableMapping[str, Any]",
        ]

        all_occurrences = []
        for pattern in dict_any_patterns:
            all_occurrences.extend(self._grep_files(python_files, pattern))

        # Filter out acceptable uses
        acceptable_files = {
            # The definition file itself is the source of truth
            "flujo/type_definitions/common.py",
            # Legacy files that still need migration
            "flujo/utils/serialization.py",
            "flujo/domain/blueprint/model_generator.py",
            "flujo/application/core/support/type_validator.py",
            # Test files are allowed to use raw dicts for fixtures/asserts
            "tests/",
        }

        concerning_uses = []
        for occurrence in all_occurrences:
            file_path = occurrence.split(":", 1)[0]

            # Skip acceptable files
            if any(acceptable in file_path for acceptable in acceptable_files):
                continue

            concerning_uses.append(occurrence)

        # Baseline for raw dict[str, Any] usages; lower this threshold as legacy debt is paid down.
        # Current baseline is ~118, we set to 125 to prevent growth.
        max_allowed_raw_dict_any = 125

        if len(concerning_uses) > max_allowed_raw_dict_any:
            pytest.fail(
                f"Found {len(concerning_uses)} uses of raw Dict/Mapping[str, Any] instead of JSONObject, "
                f"exceeding baseline of {max_allowed_raw_dict_any}.\n"
                f"Please use JSONObject from flujo.type_definitions.common instead:\n"
                + "\n".join(concerning_uses[:5])  # Show first 5
                + (f"\n... and {len(concerning_uses) - 5} more" if len(concerning_uses) > 5 else "")
            )

    def test_typed_test_fixtures_are_used(self, flujo_root: Path):
        """Verify that typed test fixtures are used in test files.

        This test ensures that new test files use the typed fixtures from
        tests/test_types/ instead of creating ad-hoc test objects.
        """
        test_files = list((flujo_root / "tests").rglob("*.py"))
        new_test_files = []

        # Find test files that might be new (simple heuristic)
        for test_file in test_files:
            try:
                with open(test_file, "r") as f:
                    content = f.read()
                    # Skip if it already uses typed fixtures
                    if (
                        "tests.test_types.fixtures" in content
                        or "tests.test_types.mocks" in content
                    ):
                        continue
                    # Flag files that create Step, StepResult, etc. without imports
                    if (
                        "Step(" in content or "StepResult(" in content
                    ) and "test_types" not in content:
                        new_test_files.append(test_file)
            except (UnicodeDecodeError, OSError):
                continue

        max_allowed_untyped_tests = 150  # Baseline for legacy tests creating ad-hoc fixtures

        if len(new_test_files) > max_allowed_untyped_tests:
            pytest.fail(
                f"Found {len(new_test_files)} test files that create test objects without using typed fixtures, "
                f"exceeding baseline of {max_allowed_untyped_tests}.\n"
                f"Please use typed fixtures from tests.test_types.fixtures and tests.test_types.mocks:\n"
                + "\n".join(str(f.relative_to(flujo_root)) for f in new_test_files[:5])
                + (f"\n... and {len(new_test_files) - 5} more" if len(new_test_files) > 5 else "")
            )

    def test_cast_usage_is_bounded(self, flujo_root: Path):
        """Bound the number of `typing.cast` usages to prevent new unsafe casts."""
        python_files = self._get_python_files(flujo_root)
        cast_occurrences = self._grep_files(python_files, "cast(")

        # Allowlist known legacy paths where casts remain; tighten over time.
        acceptable_paths = {
            "flujo/application/core/context/context_adapter.py",
            "flujo/application/core/executor_core.py",
            "flujo/application/core/policies/simple_policy.py",
            "flujo/application/core/runtime/background_task_manager.py",
            "flujo/application/core/context/context_manager.py",
            "flujo/application/core/policies/granular_policy.py",
        }

        filtered = []
        for occ in cast_occurrences:
            file_path = occ.split(":", 1)[0]
            if any(file_path.endswith(p) for p in acceptable_paths):
                continue
            filtered.append(occ)

        # Baseline: keep below 10 non-allowlisted casts (current baseline ~1)
        # Tightened from 50 to 10 to align with near-zero baseline and prevent regression
        if len(filtered) > 10:
            pytest.fail(
                f"Excessive typing.cast usage detected ({len(filtered)} > 10 baseline). "
                "Please replace casts with precise types/TypeGuards. "
                "Examples:\n" + "\n".join(filtered[:5])
            )


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

    def test_policy_classes_follow_naming_convention(self, flujo_root: Path):
        """Verify that policy classes follow the naming convention.

        Policy classes should be named Default{StepType}Executor.
        """
        policy_files = list((flujo_root / "flujo/application/core/step_policies.py").glob("*.py"))
        if not policy_files:  # Single file structure
            policy_files = [flujo_root / "flujo/application/core/step_policies.py"]

        violations = []
        for policy_file in policy_files:
            try:
                with open(policy_file, "r") as f:
                    content = f.read()

                    # Find class definitions
                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        if line.startswith("class ") and "Executor" in line:
                            class_name = line.split()[1].split("(")[0]
                            if not (
                                class_name.startswith("Default") and class_name.endswith("Executor")
                            ):
                                violations.append(f"{policy_file}:{i + 1}: {class_name}")

            except (UnicodeDecodeError, OSError):
                continue

        if violations:
            pytest.fail(
                f"Found {len(violations)} policy classes that don't follow naming convention.\n"
                "Policy classes should be named Default{StepType}Executor:\n"
                "\n".join(violations)
            )

    def test_executor_core_uses_policy_injection(self, flujo_root: Path):
        """Verify that ExecutorCore uses dependency injection for policies.

        The ExecutorCore constructor should accept policy instances as parameters.
        """
        executor_file = flujo_root / "flujo/application/core/executor_core.py"

        try:
            with open(executor_file, "r") as f:
                content = f.read()

                # Check for policy injection in __init__
                init_start = content.find("def __init__(")
                if init_start == -1:
                    pytest.fail("Could not find ExecutorCore.__init__ method")

                # Look for policy parameters in __init__
                init_end = content.find("\n    def ", init_start + 1)
                if init_end == -1:
                    init_end = len(content)

                init_content = content[init_start:init_end]

                # Check for policy injection patterns
                policy_patterns = [
                    "loop_step_executor",
                    "parallel_step_executor",
                    "conditional_step_executor",
                    "agent_step_executor",
                    "cache_step_executor",
                    "hitl_step_executor",
                    "dynamic_router_step_executor",
                ]

                missing_policies = []
                for pattern in policy_patterns:
                    if pattern not in init_content:
                        missing_policies.append(pattern)

                if missing_policies:
                    pytest.fail(
                        f"ExecutorCore.__init__ is missing policy injection for: {missing_policies}\n"
                        "All policies should be injected via constructor parameters for testability and flexibility."
                    )

        except (UnicodeDecodeError, OSError) as e:
            pytest.fail(f"Could not read executor_core.py: {e}")

    def test_exception_handling_follows_patterns(self, flujo_root: Path):
        """Verify that exception handling follows the established patterns.

        Control flow exceptions should be re-raised, not converted to failures.
        """
        policy_files = [flujo_root / "flujo/application/core/step_policies.py"]

        violations = []
        for policy_file in policy_files:
            try:
                with open(policy_file, "r") as f:
                    content = f.read()

                    # Look for patterns where control flow exceptions are caught and converted
                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        line_content = line.strip()
                        # Look for problematic patterns
                        if (
                            "except" in line_content
                            and (
                                "PausedException" in line_content
                                or "PipelineAbortSignal" in line_content
                                or "InfiniteRedirectError" in line_content
                            )
                            and "return StepResult" in lines[min(i + 1, len(lines) - 1)]
                        ):
                            violations.append(
                                f"{policy_file}:{i + 1}: Control flow exception converted to failure"
                            )

            except (UnicodeDecodeError, OSError):
                continue

        if violations:
            pytest.fail(
                f"Found {len(violations)} violations of exception handling patterns.\n"
                f"Control flow exceptions (PausedException, PipelineAbortSignal, InfiniteRedirectError) "
                f"should be re-raised, not converted to StepResult failures:\n"
                + "\n".join(violations)
            )

    def test_context_isolation_is_used_in_complex_steps(self, flujo_root: Path):
        """Verify that complex steps (LoopStep, ParallelStep) use context isolation.

        Complex steps should use ContextManager.isolate() for idempotency.
        """
        policy_files = [flujo_root / "flujo/application/core/step_policies.py"]
        policy_files.extend((flujo_root / "flujo/application/core/policies").glob("*.py"))

        violations = []
        for policy_file in policy_files:
            try:
                with open(policy_file, "r") as f:
                    content = f.read()

                    # Check for complex step classes
                    defines_complex_executor = (
                        "class LoopStepExecutor" in content
                        or "class ParallelStepExecutor" in content
                        or "class DefaultLoopStepExecutor" in content
                        or "class DefaultParallelStepExecutor" in content
                    )
                    if defines_complex_executor:
                        if "ContextManager.isolate" not in content:
                            violations.append(
                                f"{policy_file}: Complex step executor missing ContextManager.isolate()"
                            )

            except (UnicodeDecodeError, OSError):
                continue

        if violations:
            pytest.fail(
                f"Found {len(violations)} complex step executors that don't use context isolation.\n"
                f"LoopStep and ParallelStep executors must use ContextManager.isolate() for idempotency:\n"
                + "\n".join(violations)
            )

    def test_no_monolith_files(self, flujo_root: Path):
        """Verify that files are not monoliths (excessive line counts).

        File size guidelines:
        - Ideal: <= 500 lines
        - Warning: > 1000 lines (acceptable but should be refactored)
        - Failure: > 1200 lines (must be refactored)

        This test ensures code maintainability by preventing monolithic files.
        """
        python_files = self._get_python_files(flujo_root, package="flujo")

        # Exclude certain files that may legitimately be large
        excluded_files = {
            "flujo/__init__.py",  # Package init files can be large
            "flujo/type_definitions/__init__.py",  # Type definition aggregators
            "flujo/state/backends/sqlite_core.py",  # Legacy backend file
            "flujo/state/backends/postgres.py",  # Comprehensive async state API implementation
        }

        warnings = []  # Files > 1000 lines
        failures = []  # Files > 1200 lines

        for file_path in python_files:
            relative_path = str(file_path.relative_to(flujo_root))

            # Skip excluded files
            if any(excluded in relative_path for excluded in excluded_files):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    line_count = len(f.readlines())

                if line_count > 1200:
                    failures.append(f"{relative_path}: {line_count} lines")
                elif line_count > 1000:
                    warnings.append(f"{relative_path}: {line_count} lines")

            except (UnicodeDecodeError, OSError):
                continue

        # Sort by line count (descending) for better visibility
        failures.sort(key=lambda x: int(x.split(": ")[1].split()[0]), reverse=True)
        warnings.sort(key=lambda x: int(x.split(": ")[1].split()[0]), reverse=True)

        # Build failure message for files > 1200 lines
        if failures:
            error_message = (
                f"Found {len(failures)} monolith files (> 1200 lines) that MUST be refactored:\n"
                + "\n".join(failures[:10])  # Show first 10
                + (f"\n... and {len(failures) - 10} more" if len(failures) > 10 else "")
            )

            # Add warnings to the error message for context
            if warnings:
                error_message += (
                    f"\n\n⚠️  Also found {len(warnings)} large files (> 1000 lines) that should be refactored:\n"
                    + "\n".join(warnings[:5])  # Show first 5 warnings
                    + (f"\n... and {len(warnings) - 5} more" if len(warnings) > 5 else "")
                )

            pytest.fail(error_message)

        # Log warnings but don't fail (non-blocking)
        if warnings:
            warning_message = (
                f"⚠️  Warning: Found {len(warnings)} large files (> 1000 lines) that should be refactored:\n"
                + "\n".join(warnings[:5])  # Show first 5 warnings
                + (f"\n... and {len(warnings) - 5} more" if len(warnings) > 5 else "")
            )
            # Print warning to stdout (will be visible in test output)
            print(f"\n{warning_message}\n")


class TestCodeQualityGates:
    """Test suite for code quality gates."""

    @pytest.fixture
    def flujo_root(self) -> Path:
        """Get the root directory of the Flujo project."""
        return Path(__file__).parent.parent.parent

    @pytest.mark.slow
    @pytest.mark.serial
    @pytest.mark.timeout(MAKE_ALL_TIMEOUT_SECONDS)
    def test_make_all_passes(self, flujo_root: Path):
        """Verify that `make all` passes with zero errors.

        This is the primary quality gate that must pass before any PR is merged.
        """
        try:
            # Run make all from the project root
            env = os.environ.copy()
            env["SKIP_ARCHITECTURE_TESTS"] = "1"
            env["FAST_ALL"] = "1"
            make_cmd = ["make", "all"]
            if os.getenv("GITHUB_ACTIONS") == "true":
                # Avoid re-running the full fast suite inside the Architecture Tests job.
                # The PR workflow already runs Fast Tests + Unit Tests as separate jobs.
                make_cmd.append("TEST_GATE_TARGET=test-quick-check")
            summaries_before = set((flujo_root / "output").glob(FAILURE_SUMMARY_GLOB))
            result = subprocess.run(
                make_cmd,
                cwd=flujo_root,
                capture_output=True,
                text=True,
                timeout=MAKE_ALL_TIMEOUT_SECONDS,
                env=env,
            )

            if result.returncode != 0:
                # Get the last 50 lines of output for context
                output_lines = (result.stdout + result.stderr).split("\n")
                recent_output = "\n".join(output_lines[-50:])

                # Capture the freshest failure summary if available
                latest_summary = None
                try:
                    summaries_after = set((flujo_root / "output").glob(FAILURE_SUMMARY_GLOB))
                    new_summaries = summaries_after - summaries_before
                    candidates = sorted(
                        new_summaries or summaries_after,
                        key=lambda path: path.stat().st_mtime,
                        reverse=True,
                    )
                    latest_summary = candidates[0] if candidates else None
                except Exception:
                    latest_summary = None

                summary_excerpt = ""
                if latest_summary:
                    try:
                        summary_excerpt = (
                            f"\nLatest failure summary ({latest_summary}):\n"
                            f"{latest_summary.read_text()[:2000]}"
                        )
                    except Exception:
                        summary_excerpt = (
                            f"\nLatest failure summary ({latest_summary}) could not be read."
                        )

                pytest.fail(
                    f"`make all` failed with exit code {result.returncode}.\n"
                    f"This is a critical quality gate that must pass before merging.\n"
                    f"Recent output:\n{recent_output}"
                    f"{summary_excerpt}"
                )

        except subprocess.TimeoutExpired:
            pytest.fail("`make all` timed out after 15 minutes")
        except FileNotFoundError:
            pytest.fail("`make` command not found. Ensure you're in the correct environment.")

    def test_no_type_ignore_comments_without_justification(self, flujo_root: Path):
        """Verify that `# type: ignore` comments are justified.

        Type ignore comments should include a brief justification.
        """
        python_files = []
        for pattern in ["flujo/**/*.py", "tests/**/*.py"]:
            python_files.extend(list(flujo_root.glob(pattern)))

        violations = []
        for file_path in python_files:
            try:
                with open(file_path, "r") as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        if "# type: ignore" in line:
                            # Check if there's a justification comment
                            line_content = line.strip()
                            if not (
                                "#" in line_content
                                and len(line_content.split("# type: ignore")[1].strip()) > 5
                            ):
                                violations.append(f"{file_path}:{i + 1}: {line_content}")

            except (UnicodeDecodeError, OSError):
                continue

        max_allowed_violations = 50  # Track legacy ignores; tighten as fixes land

        if len(violations) > max_allowed_violations:
            pytest.fail(
                f"Found {len(violations)} # type: ignore comments without justification, "
                f"exceeding baseline of {max_allowed_violations}.\n"
                f"All type ignore comments must include a brief explanation:\n"
                + "\n".join(violations[:10])  # Show first 10
                + (f"\n... and {len(violations) - 10} more" if len(violations) > 10 else "")
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
