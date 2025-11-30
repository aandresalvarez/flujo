"""Semantic analysis for Single Responsibility Principle compliance.

This module analyzes method names and class structures to detect
when classes might be handling multiple responsibilities.
"""

import ast
import re
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import pytest


class SRPSemanticAnalyzer:
    """Analyzer for detecting SRP violations through semantic analysis."""

    # Common responsibility categories based on method naming patterns
    RESPONSIBILITY_PATTERNS = {
        "data_access": [
            r"^get_",
            r"^set_",
            r"^save_",
            r"^load_",
            r"^delete_",
            r"^update_",
            r"^insert_",
            r"^select_",
            r"^query_",
            r"^fetch_",
            r"^persist_",
            r"^store_",
            r"^retrieve_",
        ],
        "validation": [
            r"^validate_",
            r"^check_",
            r"^verify_",
            r"^is_valid",
            r"^ensure_",
            r"^confirm_",
            r"^assert_",
        ],
        "processing": [
            r"^process_",
            r"^handle_",
            r"^execute_",
            r"^run_",
            r"^perform_",
            r"^do_",
            r"^apply_",
            r"^transform_",
        ],
        "formatting": [
            r"^format_",
            r"^serialize_",
            r"^deserialize_",
            r"^parse_",
            r"^convert_",
            r"^encode_",
            r"^decode_",
        ],
        "ui_presentation": [r"^render_", r"^display_", r"^show_", r"^print_", r"^draw_"],
        "communication": [
            r"^send_",
            r"^receive_",
            r"^connect_",
            r"^disconnect_",
            r"^notify_",
            r"^publish_",
            r"^subscribe_",
        ],
        "configuration": [r"^configure_", r"^setup_", r"^initialize_", r"^init_"],
        "calculation": [
            r"^calculate_",
            r"^compute_",
            r"^sum_",
            r"^count_",
            r"^average_",
            r"^total_",
            r"^aggregate_",
        ],
    }

    def analyze_class_responsibilities(self, class_node: ast.ClassDef) -> Dict[str, int]:
        """Analyze what responsibilities a class handles based on method names."""
        responsibilities = defaultdict(int)
        method_names = []

        # Collect method names
        for item in class_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_names.append(item.name)

        # Categorize methods by responsibility
        for method_name in method_names:
            for resp_category, patterns in self.RESPONSIBILITY_PATTERNS.items():
                for pattern in patterns:
                    if re.match(pattern, method_name, re.IGNORECASE):
                        responsibilities[resp_category] += 1
                        break

        return dict(responsibilities)

    def detect_srp_violations(
        self, responsibilities: Dict[str, int], min_methods_for_analysis: int = 3
    ) -> List[str]:
        """Detect potential SRP violations based on responsibility distribution."""
        violations = []
        total_responsible_methods = sum(responsibilities.values())

        if total_responsible_methods < min_methods_for_analysis:
            return violations  # Not enough methods to analyze

        # Calculate responsibility concentration
        if responsibilities:
            max_responsibility_count = max(responsibilities.values())
            concentration_ratio = max_responsibility_count / total_responsible_methods

            # If less than 70% of methods focus on the primary responsibility,
            # it might indicate multiple responsibilities
            if concentration_ratio < 0.7:
                primary_resp = max(responsibilities.keys(), key=lambda k: responsibilities[k])
                violations.append(
                    f"Low responsibility concentration ({concentration_ratio:.2f}). "
                    f"Primary: {primary_resp} ({responsibilities[primary_resp]}/{total_responsible_methods} methods)"
                )

            # If a class handles more than 3 different types of responsibilities,
            # it's likely violating SRP
            if len(responsibilities) > 3:
                resp_list = ", ".join(f"{k}({v})" for k, v in responsibilities.items())
                violations.append(f"Multiple responsibilities detected: {resp_list}")

        return violations


class TestSRPSemanticCompliance:
    """Test suite for semantic SRP compliance."""

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

    def test_classes_have_focused_responsibilities(self, flujo_root: Path):
        """Verify that classes focus on single responsibilities based on method naming."""
        python_files = self._get_python_files(flujo_root)
        analyzer = SRPSemanticAnalyzer()
        violations = []

        for file_path in python_files:
            try:
                tree = self._parse_python_file(file_path)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        responsibilities = analyzer.analyze_class_responsibilities(node)
                        srp_violations = analyzer.detect_srp_violations(responsibilities)

                        for violation in srp_violations:
                            violations.append(f"{file_path}:{node.name}: {violation}")

            except (SyntaxError, UnicodeDecodeError, OSError):
                continue

        max_allowed_violations = 50  # Baseline for semantic analysis

        if len(violations) > max_allowed_violations:
            pytest.fail(
                f"Found {len(violations)} semantic SRP violations, exceeding baseline of {max_allowed_violations}.\n"
                f"Classes should focus on single responsibilities. First 10 violations:\n"
                + "\n".join(violations[:10])
            )

    def test_no_cross_concern_class_names(self, flujo_root: Path):
        """Verify that class names don't indicate multiple concerns."""
        python_files = self._get_python_files(flujo_root)

        # Patterns that suggest multiple responsibilities in class names
        cross_concern_patterns = [
            r".*Manager.*Controller.*",
            r".*Service.*Manager.*",
            r".*Handler.*Processor.*",
            r".*Validator.*Serializer.*",
            r".*Client.*Server.*",
            r".*Reader.*Writer.*",
            r".*Parser.*Formatter.*",
        ]

        violations = []

        for file_path in python_files:
            try:
                tree = self._parse_python_file(file_path)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_name = node.name

                        for pattern in cross_concern_patterns:
                            if re.search(pattern, class_name, re.IGNORECASE):
                                violations.append(
                                    f"{file_path}:{class_name}: Class name suggests multiple concerns"
                                )
                                break

            except (SyntaxError, UnicodeDecodeError, OSError):
                continue

        if violations:
            pytest.fail(
                f"Found {len(violations)} classes with names suggesting multiple concerns:\n"
                + "\n".join(violations)
                + "\n\nClass names should reflect single responsibilities."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
