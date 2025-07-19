#!/usr/bin/env python3
"""
Robust dependency installation script for Flujo.

This script ensures all dependencies are properly installed and provides
clear error messages for missing dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Any


def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
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


def install_dependencies(extras: List[str] = None) -> None:
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
        cmd.extend(["--extra", ",".join(extras)])

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

    print("\n✅ Installation verification complete!")


def run_tests() -> None:
    """Run a quick test to ensure everything works."""
    print("🧪 Running quick test...")

    try:
        # Test basic import
        import flujo
        print("✅ Basic import works")

        # Test core imports
        from flujo import Pipeline, step
        print("✅ Core imports work")

        # Test that we can create a simple step
        @step
        def hello(name: str) -> str:
            return f"Hello, {name}!"

        print("✅ Step creation works")

    except Exception as e:
        print(f"❌ Test failed: {e}")
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
