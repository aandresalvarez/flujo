#!/usr/bin/env python3
"""
Legacy Function Usage Analysis Tool

This tool analyzes the usage of legacy functions in the Flujo codebase
to help with the legacy cleanup process outlined in FSD_LEGACY_STEP_LOGIC_CLEANUP.md.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, Set, List, Any, Optional
from collections import defaultdict
import argparse


class LegacyFunctionAnalyzer:
    """Analyze legacy function usage in the codebase."""

    def __init__(self, codebase_path: str = "."):
        self.codebase_path = Path(codebase_path)
        self.function_usage: Dict[str, Set[str]] = defaultdict(set)
        self.import_dependencies: Dict[str, List[str]] = defaultdict(list)
        self.legacy_functions = {
            # Migrated functions (should be removed)
            '_execute_loop_step_logic',
            '_execute_conditional_step_logic',
            '_execute_parallel_step_logic',
            '_execute_dynamic_router_step_logic',

            # Deprecated functions (should emit warnings)
            '_handle_cache_step',
            '_handle_hitl_step',
            '_run_step_logic',
        }

        self.migrated_functions = {
            '_execute_loop_step_logic',
            '_execute_conditional_step_logic',
            '_execute_parallel_step_logic',
            '_execute_dynamic_router_step_logic',
        }

        self.deprecated_functions = {
            '_handle_cache_step',
            '_handle_hitl_step',
            '_run_step_logic',
        }

    def analyze_function_usage(self) -> Dict[str, Set[str]]:
        """Analyze which legacy functions are still being used."""
        print("🔍 Analyzing legacy function usage...")

        for file_path in self._find_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                tree = ast.parse(content)
                visitor = FunctionUsageVisitor(self.legacy_functions)
                visitor.visit(tree)

                if visitor.found_functions:
                    relative_path = file_path.relative_to(self.codebase_path)
                    for func_name in visitor.found_functions:
                        self.function_usage[func_name].add(str(relative_path))

            except Exception as e:
                print(f"⚠️  Error analyzing {file_path}: {e}")

        return dict(self.function_usage)

    def find_import_dependencies(self) -> Dict[str, List[str]]:
        """Find all import dependencies on legacy functions."""
        print("📦 Analyzing import dependencies...")

        for file_path in self._find_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                tree = ast.parse(content)
                visitor = ImportDependencyVisitor()
                visitor.visit(tree)

                if visitor.imports:
                    relative_path = file_path.relative_to(self.codebase_path)
                    for import_name in visitor.imports:
                        self.import_dependencies[import_name].append(str(relative_path))

            except Exception as e:
                print(f"⚠️  Error analyzing imports in {file_path}: {e}")

        return dict(self.import_dependencies)

    def generate_cleanup_report(self) -> str:
        """Generate a comprehensive cleanup report."""
        print("📊 Generating cleanup report...")

        usage_analysis = self.analyze_function_usage()
        import_analysis = self.find_import_dependencies()

        report = []
        report.append("# Legacy Function Cleanup Analysis Report")
        report.append("")
        report.append("## Summary")
        report.append("")

        # Count functions by status
        migrated_count = len([f for f in usage_analysis.keys() if f in self.migrated_functions])
        deprecated_count = len([f for f in usage_analysis.keys() if f in self.deprecated_functions])
        total_usage = sum(len(files) for files in usage_analysis.values())

        report.append(f"- **Migrated Functions Found**: {migrated_count}")
        report.append(f"- **Deprecated Functions Found**: {deprecated_count}")
        report.append(f"- **Total Usage Locations**: {total_usage}")
        report.append("")

        # Detailed analysis
        report.append("## Migrated Functions (Should Be Removed)")
        report.append("")

        for func_name in sorted(self.migrated_functions):
            if func_name in usage_analysis:
                files = usage_analysis[func_name]
                report.append(f"### {func_name}")
                report.append(f"**Status**: ❌ Still in use ({len(files)} files)")
                report.append("**Files**:")
                for file_path in sorted(files):
                    report.append(f"- `{file_path}`")
                report.append("")
            else:
                report.append(f"### {func_name}")
                report.append("**Status**: ✅ Not found in codebase")
                report.append("")

        report.append("## Deprecated Functions (Should Emit Warnings)")
        report.append("")

        for func_name in sorted(self.deprecated_functions):
            if func_name in usage_analysis:
                files = usage_analysis[func_name]
                report.append(f"### {func_name}")
                report.append(f"**Status**: ⚠️ In use ({len(files)} files)")
                report.append("**Files**:")
                for file_path in sorted(files):
                    report.append(f"- `{file_path}`")
                report.append("")
            else:
                report.append(f"### {func_name}")
                report.append("**Status**: ✅ Not found in codebase")
                report.append("")

        # Import analysis
        report.append("## Import Dependencies")
        report.append("")

        for import_name, files in sorted(import_analysis.items()):
            if any(func in import_name for func in self.legacy_functions):
                report.append(f"### {import_name}")
                report.append(f"**Files**: {len(files)}")
                for file_path in sorted(files):
                    report.append(f"- `{file_path}`")
                report.append("")

        # Recommendations
        report.append("## Recommendations")
        report.append("")

        if migrated_count > 0:
            report.append("❌ **CRITICAL**: Migrated functions are still in use!")
            report.append("These functions should have been removed. Check the files listed above.")
            report.append("")

        if deprecated_count > 0:
            report.append("⚠️ **WARNING**: Deprecated functions are still in use.")
            report.append("These functions should emit deprecation warnings.")
            report.append("")

        if total_usage == 0:
            report.append("✅ **SUCCESS**: No legacy functions found in use!")
            report.append("The cleanup appears to be complete.")
            report.append("")

        return "\n".join(report)

    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the codebase."""
        python_files = []

        for root, dirs, files in os.walk(self.codebase_path):
            # Skip common directories that shouldn't be analyzed
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', '.venv']]

            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)

        return python_files


class FunctionUsageVisitor(ast.NodeVisitor):
    """AST visitor to find function usage."""

    def __init__(self, target_functions: Set[str]):
        self.target_functions = target_functions
        self.found_functions: Set[str] = set()

    def visit_Call(self, node: ast.Call):
        """Visit function calls."""
        if isinstance(node.func, ast.Name):
            if node.func.id in self.target_functions:
                self.found_functions.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.target_functions:
                self.found_functions.add(node.func.attr)

        self.generic_visit(node)


class ImportDependencyVisitor(ast.NodeVisitor):
    """AST visitor to find import dependencies."""

    def __init__(self):
        self.imports: Set[str] = set()

    def visit_Import(self, node: ast.Import):
        """Visit import statements."""
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Visit from-import statements."""
        if node.module:
            self.imports.add(node.module)
        self.generic_visit(node)


def main():
    """Main entry point for the analysis tool."""
    parser = argparse.ArgumentParser(description="Analyze legacy function usage in Flujo codebase")
    parser.add_argument("--codebase-path", default=".", help="Path to the codebase root")
    parser.add_argument("--output", help="Output file for the report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    analyzer = LegacyFunctionAnalyzer(args.codebase_path)

    if args.verbose:
        print(f"🔍 Analyzing codebase at: {args.codebase_path}")

    # Generate the report
    report = analyzer.generate_cleanup_report()

    # Output the report
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 Report written to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
