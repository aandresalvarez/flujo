#!/usr/bin/env python3
"""
Temporary configuration file management for testing.
This script temporarily moves flujo.toml to avoid interfering with tests.
"""

import os
import shutil
import tempfile
from pathlib import Path


class TempConfigManager:
    """Manages temporary movement of flujo.toml during testing."""

    def __init__(self):
        self.config_path = Path("flujo.toml")
        self.temp_dir = None
        self.temp_path = None
        self.original_exists = False

    def __enter__(self):
        """Move flujo.toml to a temporary location."""
        if self.config_path.exists():
            self.original_exists = True
            self.temp_dir = tempfile.mkdtemp()
            self.temp_path = Path(self.temp_dir) / "flujo.toml"
            shutil.move(str(self.config_path), str(self.temp_path))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore flujo.toml from temporary location."""
        if self.original_exists:
            shutil.move(str(self.temp_path), str(self.config_path))
            shutil.rmtree(self.temp_dir)


def main():
    """Main function for command-line usage."""
    import sys

    if len(sys.argv) != 2 or sys.argv[1] not in ["hide", "show"]:
        print("Usage: python temp_config.py [hide|show]")
        sys.exit(1)

    action = sys.argv[1]
    config_path = Path("flujo.toml")

    if action == "hide":
        if config_path.exists():
            temp_dir = tempfile.mkdtemp()
            temp_path = Path(temp_dir) / "flujo.toml"
            shutil.move(str(config_path), str(temp_path))
            print(f"Hidden flujo.toml in {temp_dir}")
            # Write temp dir to a file so we can find it later
            with open(".temp_config_dir", "w") as f:
                f.write(temp_dir)
        else:
            print("flujo.toml not found")

    elif action == "show":
        if Path(".temp_config_dir").exists():
            with open(".temp_config_dir", "r") as f:
                temp_dir = f.read().strip()
            temp_path = Path(temp_dir) / "flujo.toml"
            if temp_path.exists():
                shutil.move(str(temp_path), str(config_path))
                shutil.rmtree(temp_dir)
                os.unlink(".temp_config_dir")
                print("Restored flujo.toml")
            else:
                print("Temporary flujo.toml not found")
        else:
            print("No temporary configuration found")


if __name__ == "__main__":
    main()
