# manual_testing/test_config.py
"""
Test script to demonstrate that flujo.toml configuration is working correctly.
"""

import asyncio
from flujo.infra.config_manager import get_config_manager, get_state_uri
from flujo.infra.config_manager import load_settings

async def test_configuration():
    """Test that the configuration file is being loaded correctly."""
    print("Testing Flujo Configuration")
    print("=" * 40)

    # Test 1: Check if configuration file is found
    config_manager = get_config_manager()
    config = config_manager.load_config()

    print(f"Configuration file found: {config_manager.config_path}")
    print(f"State URI from config: {config.state_uri}")

    # Test 2: Check state URI
    state_uri = get_state_uri()
    print(f"State URI from get_state_uri(): {state_uri}")

    # Test 3: Check settings
    settings = load_settings()
    print(f"Default solution model: {settings.default_solution_model}")
    print(f"Max iterations: {settings.max_iters}")

    # Test 4: Check CLI defaults
    solve_defaults = config_manager.get_cli_defaults("solve")
    print(f"Solve command defaults: {solve_defaults}")

    print("\nConfiguration test completed!")

if __name__ == "__main__":
    asyncio.run(test_configuration())
