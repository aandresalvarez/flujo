#!/usr/bin/env python3
"""
Robust dependency installation script for Flujo.

This script ensures all dependencies are properly installed and provides
clear error messages for missing dependencies.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def sanitize_command(cmd: List[str]) -> List[str]:
    """
    Sanitize command for logging to avoid exposing sensitive information.

    Args:
        cmd: The command to sanitize.

    Returns:
        List[str]: Sanitized command with sensitive parts replaced.
    """
    sanitized = []
    for part in cmd:
        # Replace potential API keys, tokens, or passwords
        if any(sensitive in part.lower() for sensitive in ['key', 'token', 'password', 'secret']):
            sanitized.append('***')
        else:
            sanitized.append(part)
    return sanitized


def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a command and return the result.

    Args:
        cmd: The command to run, provided as a list of strings.
        check: If True, the function will terminate the program with an error
            message if the command fails (non-zero return code). If False, the function
            will return the result regardless of the command's success or failure.

    Returns:
        subprocess.CompletedProcess: The result of the executed command, including
        stdout, stderr, and the return code.
    """
    sanitized_cmd = sanitize_command(cmd)
    print(f"Running: {' '.join(sanitized_cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result


def check_uv_installed() -> bool:
    """Check if uv is installed."""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_dependencies(extras: Optional[List[str]] = None) -> None:
    """Install dependencies using uv."""
    if not check_uv_installed():
        print("❌ Error: uv is not installed.")
        print("Please install uv first: https://github.com/astral-sh/uv")
        sys.exit(1)

    print("🚀 Installing dependencies with uv...")

    # Create virtual environment if it doesn't exist
    if not Path(".venv").exists():
        print("📦 Creating virtual environment...")
        run_command(["uv", "venv"])

    # Install dependencies
    cmd = ["uv", "sync"]
    if extras:
        for extra in extras:
            cmd.extend(["--extra", extra])

    run_command(cmd)
    print("✅ Dependencies installed successfully!")


def verify_installation() -> None:
    """Verify that all critical dependencies are available."""
    print("🔍 Verifying installation...")

    critical_imports = [
        "pydantic",
        "pydantic_ai",
        "pydantic_settings",
        "aiosqlite",
        "tenacity",
        "typer",
        "rich",
        "pydantic_evals",
    ]

    optional_imports = [
        "prometheus_client",
        "httpx",
        "logfire",
        "sqlvalidator",
    ]

    missing_critical = []
    missing_optional = []

    for module in critical_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            missing_critical.append(module)
            print(f"❌ {module} (CRITICAL)")

    for module in optional_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            missing_optional.append(module)
            print(f"⚠️  {module} (optional)")

    if missing_critical:
        print(f"\n❌ Critical dependencies missing: {', '.join(missing_critical)}")
        print("Please run: uv sync --all-extras")
        sys.exit(1)

    if missing_optional:
        print(f"\n⚠️  Optional dependencies missing: {', '.join(missing_optional)}")
        print("These are not required for basic functionality.")
        print("Note: sqlvalidator is used for SQL validation features but is not critical for core functionality.")

    print("\n✅ Installation verification complete!")


def run_tests() -> None:
    """Run a quick test to ensure everything works."""
    print("🧪 Running quick test...")

    try:
        # Test basic import
        import flujo  # noqa: F401
        print("✅ Basic import works")

        # Test core imports
        from flujo import step  # noqa: F401
        print("✅ Core imports work")

        # Test that we can create a simple step
        @step
        def hello(name: str) -> str:
            return f"Hello, {name}!"

        print("✅ Step creation works")

    except Exception as e:
        print(f"❌ Basic functionality test failed: {type(e).__name__}: {e}")
        print("This indicates that the installation may not be working correctly.")

        # Provide specific guidance based on exception type
        if isinstance(e, ImportError):
            print("ImportError detected. Please check:")
            print("1. That all dependencies were installed correctly")
            print("2. Try running: uv sync --all-extras")
        elif isinstance(e, ModuleNotFoundError):
            print("ModuleNotFoundError detected. Please check:")
            print("1. That you're running this from the project root directory")
            print("2. That the virtual environment is activated")
            print("3. Try running: uv sync --all-extras")
        else:
            print("Please check:")
            print("1. That all dependencies were installed correctly")
            print("2. That you're running this from the project root directory")
            print("3. That the virtual environment is activated")
            print("4. Try running: uv sync --all-extras")
        sys.exit(1)


def main():
    """Main installation function."""
    print("🔧 Flujo Dependency Installation")
    print("=" * 40)

    # Parse command line arguments
    extras = []
    if len(sys.argv) > 1:
        extras = sys.argv[1:]

    # Install dependencies
    install_dependencies(extras)

    # Verify installation
    verify_installation()

    # Run quick test
    run_tests()

    print("\n🎉 Installation complete!")
    print("\nNext steps:")
    print("1. Activate the virtual environment: source .venv/bin/activate")
    print("2. Run tests: make test")
    print("3. Run quality checks: make all")
    print("4. Try the quickstart: python examples/00_quickstart.py")


if __name__ == "__main__":
    main()
