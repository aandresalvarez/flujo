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
    """Prevent reintroduction of deprecated deprecated_serializer in core modules."""

    @pytest.fixture
    def flujo_root(self) -> Path:
        """Get the root directory of the Flujo project."""
        return Path(__file__).parent.parent.parent

    def test_no_deprecated_serializer_in_runtime_core(self, flujo_root: Path) -> None:
        """Core runtime modules must not use deprecated_serializer directly.

        deprecated_serializer is deprecated; core modules should use:
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
                if "deprecated_serializer(" in content:
                    # Skip if it's in a comment or docstring context
                    lines = content.split("\n")
                    for i, line in enumerate(lines, 1):
                        stripped = line.strip()
                        if (
                            "deprecated_serializer(" in stripped
                            and not stripped.startswith("#")
                            and not stripped.startswith('"""')
                            and not stripped.startswith("'''")
                        ):
                            rel_path = py_file.relative_to(flujo_root)
                            violations.append(f"{rel_path}:{i}: {stripped[:80]}")

        if violations:
            pytest.fail(
                f"Found deprecated_serializer usage in {len(violations)} core locations:\n"
                + "\n".join(violations[:10])
                + ("\n..." if len(violations) > 10 else "")
                + "\n\nUse model_dump(mode='json') or _serialize_for_json instead."
            )

    def test_no_robust_serializer_in_state_backends(self, flujo_root: Path) -> None:
        """Persistence backends should use the shared persistence serializer, not robust serializers."""
        backends_dir = flujo_root / "flujo/state/backends"
        if not backends_dir.exists():
            pytest.skip("state backends directory not found")

        violations: List[str] = []
        for py_file in backends_dir.rglob("*.py"):
            try:
                content = py_file.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            if "_robust_serialize_internal" in content:
                rel_path = py_file.relative_to(flujo_root)
                violations.append(str(rel_path))

        if violations:
            pytest.fail(
                "Found robust serializer usage inside persistence backends:\n"
                + "\n".join(sorted(violations))
                + "\n\nUse flujo.state.backends.base._serialize_for_json (via _fast_json_dumps) instead."
            )

    def test_no_serialize_jsonable_in_code_tests_docs(self, flujo_root: Path) -> None:
        """Legacy serialize_jsonable must not appear in project code, tests, or docs."""
        guardrail_file = Path(__file__).resolve()
        search_roots = [
            flujo_root / "flujo",
            flujo_root / "tests",
            flujo_root / "docs",
        ]

        violations: list[str] = []
        for root in search_roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_dir():
                    continue
                if path.resolve() == guardrail_file:
                    continue
                try:
                    content = path.read_text()
                except (UnicodeDecodeError, OSError):
                    continue
                if "serialize_jsonable" not in content:
                    continue
                rel_path = path.relative_to(flujo_root)
                for i, line in enumerate(content.split("\n"), 1):
                    if "serialize_jsonable" in line:
                        violations.append(f"{rel_path}:{i}: {line.strip()[:120]}")

        if violations:
            pytest.fail(
                "Found legacy serialize_jsonable references:\n"
                + "\n".join(violations[:20])
                + ("\n..." if len(violations) > 20 else "")
                + "\n\nUse model_dump(mode='json') or _serialize_for_json instead."
            )


class TestQuotaGuardrails:
    """Ensure UsageLimits→Quota translation stays centralized."""

    @pytest.fixture
    def flujo_root(self) -> Path:
        """Get the root directory of the Flujo project."""
        return Path(__file__).parent.parent.parent

    def test_limits_to_quota_boundary_is_centralized(self, flujo_root: Path) -> None:
        """UsageLimits must not be translated into Quota outside quota_manager.

        The quota system is the single enforcement surface; translation from legacy limits is a
        one-way boundary and should remain centralized to avoid divergence.
        """
        app_dir = flujo_root / "flujo/application"
        if not app_dir.exists():
            pytest.skip("Application directory not found")

        allowlist = {
            flujo_root / "flujo/application/core/runtime/quota_manager.py",
            flujo_root / "flujo/application/core/quota_manager.py",
        }
        limit_attrs = {"total_cost_usd_limit", "total_tokens_limit"}

        violations: List[str] = []
        for py_file in app_dir.rglob("*.py"):
            if py_file in allowlist:
                continue

            try:
                content = py_file.read_text()
                tree = ast.parse(content, filename=str(py_file))
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue

            has_limit_attr = any(
                isinstance(node, ast.Attribute) and node.attr in limit_attrs
                for node in ast.walk(tree)
            )
            if not has_limit_attr:
                continue

            has_quota_call = any(
                isinstance(node, ast.Call)
                and (
                    (isinstance(node.func, ast.Name) and node.func.id == "Quota")
                    or (isinstance(node.func, ast.Attribute) and node.func.attr == "Quota")
                )
                for node in ast.walk(tree)
            )
            if not has_quota_call:
                continue

            rel_path = py_file.relative_to(flujo_root)
            violations.append(str(rel_path))

        if violations:
            pytest.fail(
                "Found ad-hoc UsageLimits→Quota translation outside "
                "flujo/application/core/runtime/quota_manager.py:\n" + "\n".join(sorted(violations))
            )


class TestUsageLimitGuardrails:
    """Ensure quota remains the only enforcement surface."""

    @pytest.fixture
    def flujo_root(self) -> Path:
        """Get the root directory of the Flujo project."""
        return Path(__file__).parent.parent.parent

    def test_no_reactive_usage_limit_checks_in_core(self, flujo_root: Path) -> None:
        """Prevent reintroduction of reactive post-exec usage limit checks.

        Quota enforcement must remain proactive via Reserve→Execute→Reconcile. Core runtime code
        should not compare accumulated usage to `UsageLimits.*_limit` using <, <=, >, >=.
        """
        core_dir = flujo_root / "flujo/application/core"
        if not core_dir.exists():
            pytest.skip("core directory not found")

        allowed_files = {
            "flujo/application/core/runtime/quota_manager.py",
            "flujo/application/core/runtime/usage_messages.py",
            "flujo/application/core/quota_manager.py",
            "flujo/application/core/usage_messages.py",
        }

        violations: list[str] = []
        for py_file in core_dir.rglob("*.py"):
            rel_path = str(py_file.relative_to(flujo_root))
            if rel_path in allowed_files:
                continue
            try:
                content = py_file.read_text()
                tree = ast.parse(content, filename=str(py_file))
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.Compare):
                    continue
                if not any(isinstance(op, (ast.Gt, ast.GtE, ast.Lt, ast.LtE)) for op in node.ops):
                    continue

                referenced_attrs: set[str] = set()
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Attribute) and sub.attr in {
                        "total_cost_usd_limit",
                        "total_tokens_limit",
                    }:
                        referenced_attrs.add(sub.attr)
                if referenced_attrs:
                    line = content.split("\n")[node.lineno - 1].strip()
                    violations.append(f"{rel_path}:{node.lineno}: {line}")

        if violations:
            pytest.fail(
                "Found reactive usage limit checks in core runtime modules:\n"
                + "\n".join(violations[:20])
                + ("\n..." if len(violations) > 20 else "")
                + "\n\nUse quota reservation (Reserve→Execute→Reconcile) instead."
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
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        target = alias.name
                        if target == "flujo.application.core" or target.startswith(
                            "flujo.application.core."
                        ):
                            line = content.split("\n")[node.lineno - 1].strip()
                            rel_path = py_file.relative_to(flujo_root)
                            violations.append(f"{rel_path}:{node.lineno}: {line}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    # Prepend dots for relative imports to keep intent visible
                    if node.level:
                        module = "." * node.level + module

                    normalized = module.lstrip(".")
                    if normalized.startswith("application.core"):
                        normalized = "flujo." + normalized

                    if normalized == "flujo.application.core" or normalized.startswith(
                        "flujo.application.core."
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
        assert "def accepts_param(" in content, (
            "domain.interfaces must provide accepts_param function"
        )
        assert '"accepts_param"' in content or "'accepts_param'" in content, (
            "accepts_param must be exported in __all__"
        )


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
        assert "from" in content and "async_bridge" in content, (
            "prometheus.py should import from async_bridge"
        )
        # Should NOT define its own run_coroutine implementation
        assert "def run_coroutine(" not in content, (
            "prometheus.py should not define its own run_coroutine; use run_sync"
        )

    def test_no_ad_hoc_blocking_portal(self, flujo_root: Path) -> None:
        """Prevent duplicate `BlockingPortal` usage outside the shared bridge.

        The shared bridge is `flujo/utils/async_bridge.py`. All sync→async execution should use it.
        """
        bridge_file = flujo_root / "flujo/utils/async_bridge.py"
        if not bridge_file.exists():
            pytest.skip("async_bridge.py not found")

        violations: List[str] = []
        for py_file in (flujo_root / "flujo").rglob("*.py"):
            if py_file == bridge_file:
                continue

            try:
                content = py_file.read_text()
            except (UnicodeDecodeError, OSError):
                continue

            if "start_blocking_portal" in content or "BlockingPortal" in content:
                rel_path = py_file.relative_to(flujo_root)
                violations.append(str(rel_path))

        if violations:
            pytest.fail(
                "Found ad-hoc BlockingPortal usage outside flujo/utils/async_bridge.py:\n"
                + "\n".join(sorted(violations))
            )

    def test_no_ad_hoc_asyncio_runners(self, flujo_root: Path) -> None:
        """Prevent ad-hoc asyncio loop runners outside the shared async bridge."""
        bridge_file = flujo_root / "flujo/utils/async_bridge.py"
        violations: list[str] = []

        for py_file in (flujo_root / "flujo").rglob("*.py"):
            if py_file == bridge_file:
                continue

            try:
                content = py_file.read_text()
                tree = ast.parse(content, filename=str(py_file))
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Attribute):
                    if (
                        isinstance(func.value, ast.Name)
                        and func.value.id == "asyncio"
                        and func.attr in {"run", "new_event_loop", "run_coroutine_threadsafe"}
                    ):
                        line = content.split("\n")[node.lineno - 1].strip()
                        rel_path = py_file.relative_to(flujo_root)
                        violations.append(f"{rel_path}:{node.lineno}: {line}")
                    if func.attr == "run_until_complete":
                        line = content.split("\n")[node.lineno - 1].strip()
                        rel_path = py_file.relative_to(flujo_root)
                        violations.append(f"{rel_path}:{node.lineno}: {line}")
                elif isinstance(func, ast.Name) and func.id == "run_coroutine_threadsafe":
                    line = content.split("\n")[node.lineno - 1].strip()
                    rel_path = py_file.relative_to(flujo_root)
                    violations.append(f"{rel_path}:{node.lineno}: {line}")

        if violations:
            pytest.fail(
                "Found ad-hoc asyncio runner usage outside flujo/utils/async_bridge.py:\n"
                + "\n".join(violations[:20])
                + ("\n..." if len(violations) > 20 else "")
                + "\n\nUse flujo.utils.async_bridge.run_sync instead."
            )
