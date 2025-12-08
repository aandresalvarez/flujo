"""Architecture tests to guard serialization and import patterns.

These tests prevent reintroduction of deprecated patterns and enforce
architectural boundaries established in the remediation plan.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List

import pytest


pytestmark = [pytest.mark.slow]


class TestSerializationGuardrails:
    """Prevent reintroduction of deprecated serialize_jsonable in core modules."""

    @pytest.fixture
    def flujo_root(self) -> Path:
        """Get the root directory of the Flujo project."""
        return Path(__file__).parent.parent.parent

    def test_no_serialize_jsonable_in_runtime_core(self, flujo_root: Path) -> None:
        """Core runtime modules must not use serialize_jsonable directly.

        serialize_jsonable is deprecated; core modules should use:
        - model_dump(mode="json") for Pydantic models
        - _serialize_for_json for primitives
        """
        core_dirs = [
            flujo_root / "flujo/application/core",
            flujo_root / "flujo/state/backends",
        ]

        violations: List[str] = []
        for core_dir in core_dirs:
            if not core_dir.exists():
                continue
            for py_file in core_dir.rglob("*.py"):
                try:
                    content = py_file.read_text()
                except (UnicodeDecodeError, OSError):
                    continue

                # Check for direct usage (not just definition/export)
                if "serialize_jsonable(" in content:
                    # Skip if it's in a comment or docstring context
                    lines = content.split("\n")
                    for i, line in enumerate(lines, 1):
                        stripped = line.strip()
                        if (
                            "serialize_jsonable(" in stripped
                            and not stripped.startswith("#")
                            and not stripped.startswith('"""')
                            and not stripped.startswith("'''")
                        ):
                            rel_path = py_file.relative_to(flujo_root)
                            violations.append(f"{rel_path}:{i}: {stripped[:80]}")

        if violations:
            pytest.fail(
                f"Found serialize_jsonable usage in {len(violations)} core locations:\n"
                + "\n".join(violations[:10])
                + ("\n..." if len(violations) > 10 else "")
                + "\n\nUse model_dump(mode='json') or _serialize_for_json instead."
            )


class TestDSLCoreDecoupling:
    """Ensure DSL modules do not import from application.core at module level."""

    @pytest.fixture
    def flujo_root(self) -> Path:
        """Get the root directory of the Flujo project."""
        return Path(__file__).parent.parent.parent

    def test_dsl_no_module_level_core_imports(self, flujo_root: Path) -> None:
        """DSL modules must use lazy imports or domain.interfaces for core dependencies.

        Module-level imports from flujo.application.core create circular import
        risks and tightly couple declaration (DSL) with execution (core).

        Allowed patterns:
        - Lazy imports inside functions/methods
        - TYPE_CHECKING guard imports
        - Imports from flujo.domain.interfaces
        """
        dsl_dir = flujo_root / "flujo/domain/dsl"
        if not dsl_dir.exists():
            pytest.skip("DSL directory not found")

        violations: List[str] = []

        for py_file in dsl_dir.rglob("*.py"):
            try:
                content = py_file.read_text()
                tree = ast.parse(content, filename=str(py_file))
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue

            # Check for module-level imports from application.core
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Get the module being imported
                    if isinstance(node, ast.ImportFrom) and node.module:
                        module = node.module
                        if "flujo.application.core" in module or module.startswith(
                            "...application.core"
                        ):
                            line = content.split("\n")[node.lineno - 1].strip()
                            rel_path = py_file.relative_to(flujo_root)
                            violations.append(f"{rel_path}:{node.lineno}: {line}")

        if violations:
            pytest.fail(
                f"Found {len(violations)} module-level core imports in DSL:\n"
                + "\n".join(violations)
                + "\n\nUse lazy imports inside methods or flujo.domain.interfaces."
            )

    def test_interfaces_module_provides_accepts_param(self, flujo_root: Path) -> None:
        """Verify domain.interfaces exports accepts_param for DSL usage."""
        interfaces_file = flujo_root / "flujo/domain/interfaces.py"
        if not interfaces_file.exists():
            pytest.skip("interfaces.py not found")

        content = interfaces_file.read_text()
        assert (
            "def accepts_param(" in content
        ), "domain.interfaces must provide accepts_param function"
        assert (
            '"accepts_param"' in content or "'accepts_param'" in content
        ), "accepts_param must be exported in __all__"


class TestAsyncBridgeUnification:
    """Ensure async/sync bridge utilities are centralized."""

    @pytest.fixture
    def flujo_root(self) -> Path:
        """Get the root directory of the Flujo project."""
        return Path(__file__).parent.parent.parent

    def test_shared_async_bridge_exists(self, flujo_root: Path) -> None:
        """Verify the shared async bridge utility exists."""
        bridge_file = flujo_root / "flujo/utils/async_bridge.py"
        assert bridge_file.exists(), "flujo/utils/async_bridge.py must exist with run_sync utility"

        content = bridge_file.read_text()
        assert "def run_sync(" in content, "async_bridge.py must provide run_sync function"

    def test_prometheus_uses_shared_bridge(self, flujo_root: Path) -> None:
        """Verify prometheus.py uses the shared async bridge, not ad-hoc threading."""
        prometheus_file = flujo_root / "flujo/telemetry/prometheus.py"
        if not prometheus_file.exists():
            pytest.skip("prometheus.py not found")

        content = prometheus_file.read_text()
        # Should import from async_bridge
        assert (
            "from" in content and "async_bridge" in content
        ), "prometheus.py should import from async_bridge"
        # Should NOT define its own run_coroutine implementation
        assert (
            "def run_coroutine(" not in content
        ), "prometheus.py should not define its own run_coroutine; use run_sync"
