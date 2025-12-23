"""Single Responsibility Principle compliance tests.

These tests verify that classes follow the Single Responsibility Principle
by analyzing class size, method count, and cohesion metrics.
"""

import ast
from pathlib import Path
from typing import List, Dict
import pytest


class TestSRPCompliance:
    """Test suite for Single Responsibility Principle compliance."""

    @pytest.fixture
    def flujo_root(self) -> Path:
        """Get the root directory of the Flujo project."""
        return Path(__file__).parent.parent.parent

    def _get_python_files(self, flujo_root: Path, package: str = "flujo") -> List[Path]:
        """Get all Python files in the specified package."""
        package_dir = flujo_root / package
        return list(package_dir.rglob("*.py"))

    def _parse_python_file(self, file_path: Path) -> ast.Module:
        """Parse a Python file into an AST."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return ast.parse(content, filename=str(file_path))

    def _analyze_class_complexity(self, tree: ast.Module) -> Dict[str, Dict]:
        """Analyze classes for SRP violations."""
        classes = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "methods": [],
                    "line_count": 0,
                    "dependencies": set(),
                    "public_methods": 0,
                    "private_methods": 0,
                    "properties": 0,
                    "attributes": set(),
                }

                # Count methods and analyze method types
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if not item.name.startswith("_"):
                            class_info["public_methods"] += 1
                        else:
                            class_info["private_methods"] += 1
                        class_info["methods"].append(item.name)

                        # Analyze method body for complexity indicators
                        method_complexity = self._analyze_method_complexity(item)
                        class_info.setdefault("method_complexities", []).append(method_complexity)

                    elif isinstance(item, ast.AsyncFunctionDef):
                        if not item.name.startswith("_"):
                            class_info["public_methods"] += 1
                        else:
                            class_info["private_methods"] += 1
                        class_info["methods"].append(item.name)

                        method_complexity = self._analyze_method_complexity(item)
                        class_info.setdefault("method_complexities", []).append(method_complexity)

                # Calculate metrics
                total_methods = class_info["public_methods"] + class_info["private_methods"]
                class_info["total_methods"] = total_methods

                # SRP Indicators
                class_info["srp_violations"] = []

                # Too many public methods (God Object indicator)
                if class_info["public_methods"] > 15:
                    class_info["srp_violations"].append(
                        f"Too many public methods ({class_info['public_methods']}) - possible God Object"
                    )

                # Too many total methods
                if total_methods > 30:
                    class_info["srp_violations"].append(
                        f"Too many methods ({total_methods}) - consider splitting responsibilities"
                    )

                # High method complexity average
                if "method_complexities" in class_info:
                    avg_complexity = sum(class_info["method_complexities"]) / len(
                        class_info["method_complexities"]
                    )
                    if avg_complexity > 25:
                        class_info["srp_violations"].append(
                            f"High average method complexity ({avg_complexity:.1f}) - methods doing too much"
                        )

                classes[node.name] = class_info

        return classes

    def _analyze_method_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a method."""
        complexity = 1  # Base complexity

        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, ast.Try):
                complexity += 1

        return complexity

    def test_classes_follow_srp_method_count_limits(self, flujo_root: Path):
        """Verify that classes don't violate SRP through excessive method counts."""
        python_files = self._get_python_files(flujo_root)
        violations = []

        for file_path in python_files:
            try:
                tree = self._parse_python_file(file_path)
                classes = self._analyze_class_complexity(tree)

                for class_name, class_info in classes.items():
                    if class_info["srp_violations"]:
                        violations.extend(
                            [
                                f"{file_path}:{class_name}: {violation}"
                                for violation in class_info["srp_violations"]
                            ]
                        )

            except (SyntaxError, UnicodeDecodeError, OSError):
                continue

        # Allow some baseline violations for legacy code
        max_allowed_violations = 100

        if len(violations) > max_allowed_violations:
            pytest.fail(
                f"Found {len(violations)} SRP violations, exceeding baseline of {max_allowed_violations}.\n"
                f"Classes should follow Single Responsibility Principle. First 10 violations:\n"
                + "\n".join(violations[:10])
            )

    def test_no_god_classes_by_method_count(self, flujo_root: Path):
        """Verify no classes have excessive public methods (God Object pattern)."""
        python_files = self._get_python_files(flujo_root)
        god_classes = []

        # Abstract base classes / interfaces may legitimately have many methods
        # as they define complete API contracts
        allowed_abstract_classes = {
            "StateBackend",  # ABC defining complete state backend interface
        }

        for file_path in python_files:
            try:
                tree = self._parse_python_file(file_path)
                classes = self._analyze_class_complexity(tree)

                for class_name, class_info in classes.items():
                    # Skip allowed abstract/interface classes
                    if class_name in allowed_abstract_classes:
                        continue
                    if class_info["public_methods"] > 25:  # Very high threshold for God Objects
                        god_classes.append(
                            f"{file_path}:{class_name}: {class_info['public_methods']} public methods"
                        )

            except (SyntaxError, UnicodeDecodeError, OSError):
                continue

        if god_classes:
            pytest.fail(
                f"Found {len(god_classes)} potential God Classes with excessive public methods:\n"
                + "\n".join(god_classes)
                + "\n\nGod Classes violate SRP - consider splitting into smaller, focused classes."
            )

    def test_method_complexity_indicates_focused_responsibilities(self, flujo_root: Path):
        """Verify that methods have focused responsibilities (not too complex)."""
        python_files = self._get_python_files(flujo_root)
        complex_methods = []

        for file_path in python_files:
            try:
                tree = self._parse_python_file(file_path)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        complexity = self._analyze_method_complexity(node)
                        if complexity > 40:  # Very high complexity threshold
                            complex_methods.append(
                                f"{file_path}:{node.name}: complexity {complexity}"
                            )

            except (SyntaxError, UnicodeDecodeError, OSError):
                continue

        max_allowed_complex_methods = 100

        if len(complex_methods) > max_allowed_complex_methods:
            pytest.fail(
                f"Found {len(complex_methods)} highly complex methods, exceeding baseline of {max_allowed_complex_methods}.\n"
                f"Complex methods often indicate multiple responsibilities. Consider refactoring:\n"
                + "\n".join(complex_methods[:10])
            )

    def _analyze_class_cohesion(self, class_node: ast.ClassDef) -> Dict[str, float]:
        """Analyze class cohesion by examining method name similarity and dependencies."""
        methods = []
        method_names = []

        for item in class_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item)
                method_names.append(item.name)

        if len(methods) < 2:
            return {"cohesion_score": 1.0, "dominant_concern": "single_method"}

        # Simple semantic analysis based on method name prefixes
        prefixes = {}
        for name in method_names:
            # Remove common prefixes like 'get_', 'set_', 'is_', etc.
            clean_name = (
                name.replace("get_", "").replace("set_", "").replace("is_", "").replace("has_", "")
            )
            prefix = clean_name.split("_")[0] if "_" in clean_name else clean_name[:4]
            prefixes[prefix] = prefixes.get(prefix, 0) + 1

        # Calculate cohesion as the ratio of methods sharing the most common concern
        if prefixes:
            max_concern_count = max(prefixes.values())
            cohesion_score = max_concern_count / len(method_names)
        else:
            cohesion_score = 1.0

        dominant_concern = (
            max(prefixes.keys(), key=lambda k: prefixes[k]) if prefixes else "unknown"
        )

        return {
            "cohesion_score": cohesion_score,
            "dominant_concern": dominant_concern,
            "concern_distribution": prefixes,
        }

    def test_class_cohesion_indicates_single_responsibility(self, flujo_root: Path):
        """Verify that classes have high cohesion (methods related to same concern)."""
        python_files = self._get_python_files(flujo_root)
        low_cohesion_classes = []

        for file_path in python_files:
            try:
                tree = self._parse_python_file(file_path)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        cohesion_analysis = self._analyze_class_cohesion(node)

                        # Low cohesion indicates potential SRP violation
                        if cohesion_analysis["cohesion_score"] < 0.3 and len(node.body) > 8:
                            low_cohesion_classes.append(
                                f"{file_path}:{node.name}: cohesion {cohesion_analysis['cohesion_score']:.2f} "
                                f"(dominant: {cohesion_analysis['dominant_concern']})"
                            )

            except (SyntaxError, UnicodeDecodeError, OSError):
                continue

        max_allowed_low_cohesion = 100

        if len(low_cohesion_classes) > max_allowed_low_cohesion:
            pytest.fail(
                f"Found {len(low_cohesion_classes)} classes with low cohesion, exceeding baseline of {max_allowed_low_cohesion}.\n"
                f"Low cohesion often indicates SRP violations. Classes should focus on single concerns:\n"
                + "\n".join(low_cohesion_classes[:10])
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
