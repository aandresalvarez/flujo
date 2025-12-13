"""AST-based lint to ensure adapters are explicitly allowlisted.

This script uses Python's AST module to detect adapter class usages (instantiations,
calls, and references) and validates that the appropriate ADAPTER_ALLOW token is
present in the same file.

Usage:
    python scripts/lint_adapter_allowlist.py [--verbose]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST_PATH = ROOT / "scripts" / "adapter_allowlist.json"

# Adapter classes that require allowlist tokens
ADAPTER_CLASSES: Set[str] = {
    "DictContextAdapter",
    "SchemaAdapter",
    "TypedAny",
    "TypedAnyAdapter",
    "UnknownAdapter",
}


@dataclass
class AdapterUsage:
    """Represents a detected adapter usage in source code."""

    file_path: Path
    adapter_name: str
    line_number: int
    usage_type: str  # 'instantiation', 'call', 'reference', 'import'


class AdapterUsageVisitor(ast.NodeVisitor):
    """AST visitor that detects adapter class usages."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.usages: List[AdapterUsage] = []
        self._imported_adapters: Set[str] = set()

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track adapter imports."""
        if node.names:
            for alias in node.names:
                name = alias.name
                if name in ADAPTER_CLASSES:
                    self._imported_adapters.add(alias.asname or name)
                    self.usages.append(
                        AdapterUsage(
                            file_path=self.file_path,
                            adapter_name=name,
                            line_number=node.lineno,
                            usage_type="import",
                        )
                    )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Detect adapter instantiations and calls."""
        # 1. Check for class instantiation
        adapter_name = self._extract_adapter_name(node.func)
        if adapter_name:
            self.usages.append(
                AdapterUsage(
                    file_path=self.file_path,
                    adapter_name=adapter_name,
                    line_number=node.lineno,
                    usage_type="instantiation",
                )
            )

        # 2. Check for function calls (step, adapter_step)
        func_name = self._extract_function_name(node.func)
        if func_name in {"step", "adapter_step"}:
            self._check_step_call(node, func_name)

        self.generic_visit(node)

    def _check_step_call(self, node: ast.Call, func_name: str) -> None:
        """Check step() and adapter_step() calls for metadata enforcement."""
        kwargs = {k.arg: k.value for k in node.keywords if k.arg}

        is_adapter = False
        if func_name == "adapter_step":
            is_adapter = True
        elif func_name == "step":
            # Check is_adapter=True
            if "is_adapter" in kwargs:
                val = kwargs["is_adapter"]
                if isinstance(val, ast.Constant) and val.value is True:
                    is_adapter = True

        if is_adapter:
            # Must have adapter_id and adapter_allow
            if "adapter_id" not in kwargs or "adapter_allow" not in kwargs:
                # We can't easily append a violation here because usages tracks 'names'
                # We'll treat this as a usage of a special "MissingMetadata" adapter to flag it
                self.usages.append(
                    AdapterUsage(
                        file_path=self.file_path,
                        adapter_name="MissingMetadata",
                        line_number=node.lineno,
                        usage_type="metadata_violation",
                    )
                )
                return

            # Check adapter_allow token
            allow_node = kwargs["adapter_allow"]
            if isinstance(allow_node, ast.Constant) and isinstance(allow_node.value, str):
                token = allow_node.value
                # We record usage of this TOKEN as a pseudo-adapter so logic verify it exists
                self.usages.append(
                    AdapterUsage(
                        file_path=self.file_path,
                        adapter_name=f"TOKEN:{token}",
                        line_number=node.lineno,
                        usage_type="token_usage",
                    )
                )

    def _extract_function_name(self, node: ast.expr) -> str | None:
        """Extract function name from call."""
        if isinstance(node, ast.Name):
            return node.id
        # Handle decorators or module attributes? e.g. dsl.step
        if isinstance(node, ast.Attribute):
            # We assume simple attribute access like dsl.step
            return node.attr
        return None

    def _extract_adapter_name(self, node: ast.expr) -> str | None:
        """Extract adapter class name from a call target."""
        if isinstance(node, ast.Name):
            if node.id in ADAPTER_CLASSES or node.id in self._imported_adapters:
                return node.id
        elif isinstance(node, ast.Attribute):
            if node.attr in ADAPTER_CLASSES:
                return node.attr
        return None


def _load_allowlist() -> dict[str, str]:
    """Load the adapter allowlist from JSON."""
    if not ALLOWLIST_PATH.exists():
        sys.stderr.write(
            "Adapter allowlist missing. Commit scripts/adapter_allowlist.json to proceed.\n"
        )
        sys.exit(1)
    try:
        loaded = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"Unable to read/parse adapter allowlist: {exc}\n")
        sys.exit(1)
    if not isinstance(loaded, dict):
        sys.stderr.write("Adapter allowlist must be a JSON object at the top level.\n")
        sys.exit(1)
    allowed = loaded.get("allowed")
    if not isinstance(allowed, dict):
        sys.stderr.write("Adapter allowlist must include an 'allowed' JSON object.\n")
        sys.exit(1)
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in allowed.items()):
        sys.stderr.write("Adapter allowlist 'allowed' values must be string-to-string mapping.\n")
        sys.exit(1)
    return allowed


def _find_adapter_usages_ast(verbose: bool = False) -> List[AdapterUsage]:
    """Find adapter usages using AST parsing."""
    all_usages: List[AdapterUsage] = []
    flujo_dir = ROOT / "flujo"

    for file_path in flujo_dir.rglob("*.py"):
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))
            visitor = AdapterUsageVisitor(file_path)
            visitor.visit(tree)
            all_usages.extend(visitor.usages)
            if verbose and visitor.usages:
                print(f"  Found {len(visitor.usages)} usages in {file_path.relative_to(ROOT)}")
        except SyntaxError as e:
            if verbose:
                print(f"  Skipping {file_path.relative_to(ROOT)}: {e}")
        except (OSError, UnicodeDecodeError, ValueError) as e:
            if verbose:
                print(f"  Error parsing {file_path.relative_to(ROOT)}: {e}")

    return all_usages


def _has_token(source: str, token: str) -> bool:
    """Check if the allowlist token is present in the source."""
    return f"ADAPTER_ALLOW:{token}" in source


def _deduplicate_usages(usages: List[AdapterUsage]) -> Dict[Path, Set[str]]:
    """Group and deduplicate usages by file and adapter name."""
    file_adapters: Dict[Path, Set[str]] = {}
    for usage in usages:
        if usage.file_path not in file_adapters:
            file_adapters[usage.file_path] = set()
        file_adapters[usage.file_path].add(usage.adapter_name)
    return file_adapters


def main() -> None:
    parser = argparse.ArgumentParser(description="AST-based adapter allowlist lint.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    if args.verbose:
        print("Scanning for adapter usages using AST parsing...")

    allowlist = _load_allowlist()
    usages = _find_adapter_usages_ast(verbose=args.verbose)

    if args.verbose:
        print(f"\nFound {len(usages)} total adapter usages")
        for adapter in ADAPTER_CLASSES:
            count = sum(1 for u in usages if u.adapter_name == adapter)
            if count > 0:
                print(f"  {adapter}: {count} usages")
        print()

    # Deduplicate: we only need to check once per (file, adapter) pair
    file_adapters = _deduplicate_usages(usages)

    violations: List[str] = []
    for file_path, adapters in file_adapters.items():
        try:
            source = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            violations.append(
                f"Unable to read {file_path.relative_to(ROOT)} to verify allowlist token: {exc}"
            )
            continue

        for adapter in adapters:
            if adapter == "MissingMetadata":
                line_nums = [
                    u.line_number
                    for u in usages
                    if u.file_path == file_path and u.adapter_name == "MissingMetadata"
                ]
                line_info = f" (line {line_nums[0]})" if line_nums else ""
                violations.append(
                    f"Adapter definition in {file_path.relative_to(ROOT)}{line_info} missing explicit 'adapter_id' and 'adapter_allow' metadata."
                )
                continue

            if adapter.startswith("TOKEN:"):
                token = adapter.split(":", 1)[1]
                if not _has_token(source, token):
                    line_nums = [
                        u.line_number
                        for u in usages
                        if u.file_path == file_path and u.adapter_name == adapter
                    ]
                    line_info = f" (line {line_nums[0]})" if line_nums else ""
                    violations.append(
                        f"Adapter usage in {file_path.relative_to(ROOT)}{line_info} references token '{token}' but ADAPTER_ALLOW:{token} is missing from source."
                    )
                continue

            token = allowlist.get(adapter)
            if token is None:
                # Find line number of first usage for better error reporting
                line_nums = [
                    u.line_number
                    for u in usages
                    if u.file_path == file_path and u.adapter_name == adapter
                ]
                line_info = f" (line {line_nums[0]})" if line_nums else ""
                violations.append(
                    f"{adapter} used in {file_path.relative_to(ROOT)}{line_info} without allowlist entry"
                )
                continue

            if not _has_token(source, token):
                line_nums = [
                    u.line_number
                    for u in usages
                    if u.file_path == file_path and u.adapter_name == adapter
                ]
                line_info = f" (line {line_nums[0]})" if line_nums else ""
                violations.append(
                    f"{adapter} in {file_path.relative_to(ROOT)}{line_info} missing token ADAPTER_ALLOW:{token}"
                )

    if violations:
        sys.stderr.write("Adapter allowlist violations detected:\n")
        for violation in violations[:20]:  # Cap output at 20 violations
            sys.stderr.write(f" - {violation}\n")
        if len(violations) > 20:
            sys.stderr.write(f" ... and {len(violations) - 20} more violations\n")
        sys.exit(1)

    print("✅ Adapter allowlist lint passed (AST-based, no unapproved adapter usage found).")


if __name__ == "__main__":
    main()
