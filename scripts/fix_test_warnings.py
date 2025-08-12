#!/usr/bin/env python3
"""
Script to fix test warnings by replacing direct Flujo instantiation with create_test_flujo.
"""

import re
import sys
from pathlib import Path


def fix_test_file(file_path: Path) -> bool:
    """Fix a single test file by replacing Flujo() calls with create_test_flujo()."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Add import if not present
        if "from tests.conftest import create_test_flujo" not in content:
            # Find the last import line
            import_pattern = r"^(from flujo.*import.*)$"
            matches = list(re.finditer(import_pattern, content, re.MULTILINE))
            if matches:
                last_import = matches[-1]
                # Insert our import after the last flujo import
                content = (
                    content[: last_import.end()]
                    + "\nfrom tests.conftest import create_test_flujo"
                    + content[last_import.end() :]
                )
            else:
                # If no flujo imports found, add at the top after existing imports
                lines = content.split("\n")
                import_end = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith(("import ", "from ")):
                        import_end = i + 1
                    elif line.strip() and not line.startswith("#"):
                        break

                lines.insert(import_end, "from tests.conftest import create_test_flujo")
                content = "\n".join(lines)

        # Replace Flujo( with create_test_flujo(
        # But be careful not to replace imports or other contexts
        lines = content.split("\n")
        modified = False

        for i, line in enumerate(lines):
            # Skip import lines and lines that are already using create_test_flujo
            if ("import" in line and "Flujo" in line) or "create_test_flujo" in line:
                continue

            # Replace Flujo( with create_test_flujo( but only in function calls
            if "Flujo(" in line and not line.strip().startswith("#"):
                # Make sure this is actually a function call, not an import or assignment
                if not any(
                    skip in line
                    for skip in ["import Flujo", "from flujo import", "Flujo,", "Flujo["]
                ):
                    new_line = line.replace("Flujo(", "create_test_flujo(")
                    if new_line != line:
                        lines[i] = new_line
                        modified = True

        if modified:
            content = "\n".join(lines)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main function to fix all test files."""
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"

    if not tests_dir.exists():
        print("Tests directory not found!")
        sys.exit(1)

    # Find all Python test files
    test_files = []
    for pattern in ["**/*.py"]:
        test_files.extend(tests_dir.glob(pattern))

    # Filter out conftest.py and __init__.py
    test_files = [f for f in test_files if f.name not in ["conftest.py", "__init__.py"]]

    print(f"Found {len(test_files)} test files to process...")

    fixed_count = 0
    for test_file in test_files:
        if fix_test_file(test_file):
            print(f"Fixed: {test_file}")
            fixed_count += 1

    print(f"\nFixed {fixed_count} test files.")
    print("Run 'make test' to verify the fixes work correctly.")


if __name__ == "__main__":
    main()
