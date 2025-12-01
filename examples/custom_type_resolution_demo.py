#!/usr/bin/env python3
"""
Demonstration of the new generic type resolution system.

This example shows how to:
1. Register custom types using the public API
2. Use automatic type discovery
3. Handle complex type scenarios
4. Benefit from the security and performance improvements
"""

import asyncio
from typing import Optional, List
from pydantic import BaseModel

from flujo.application.core.context_adapter import register_custom_type
from flujo.domain.models import BaseModel as FlujoBaseModel
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step
from flujo.type_definitions.common import JSONObject


# Define custom types that users might create
class UserProfile(BaseModel):
    """User profile model."""

    name: str
    email: str
    age: int
    is_active: bool = True


class UserSettings(BaseModel):
    """User settings model."""

    theme: str = "dark"
    notifications: bool = True
    language: str = "en"


class UserPreferences(BaseModel):
    """User preferences model."""

    favorite_colors: List[str] = []
    privacy_level: str = "medium"
    auto_save: bool = True


class ComplexUserData(BaseModel):
    """Complex user data with nested models."""

    profile: UserProfile
    settings: Optional[UserSettings] = None
    preferences: UserPreferences
    metadata: JSONObject = {}


# Register custom types using the new public API
def register_custom_types():
    """Register all custom types for type resolution."""
    register_custom_type(UserProfile)
    register_custom_type(UserSettings)
    register_custom_type(UserPreferences)
    register_custom_type(ComplexUserData)

    print("✅ Custom types registered successfully")


# Define a context model that uses the custom types
class UserContext(FlujoBaseModel):
    """Context model that uses custom types."""

    current_user: UserProfile
    user_settings: Optional[UserSettings] = None
    user_preferences: UserPreferences
    complex_data: ComplexUserData


# Example step that processes user data
async def process_user_data(data: JSONObject, context: UserContext) -> JSONObject:
    """Process user data with custom type resolution."""
    print(f"Processing user data for: {data.get('name', 'Unknown')}")

    # The type resolution system will automatically handle the custom types
    # because they're registered and discovered from the context model

    return {
        "processed": True,
        "user_name": data.get("name"),
        "settings_theme": context.user_settings.theme if context.user_settings else "default",
        "preferences_count": len(context.user_preferences.favorite_colors),
    }


# Example step that updates user settings
async def update_user_settings(data: JSONObject, context: UserContext) -> UserSettings:
    """Update user settings with automatic type resolution."""
    print("Updating user settings...")

    # Create new settings from the input data
    new_settings = UserSettings(
        theme=data.get("theme", "light"),
        notifications=data.get("notifications", True),
        language=data.get("language", "en"),
    )

    # The type resolution system will handle UserSettings automatically
    return new_settings


async def main():
    """Main demonstration function."""
    print("🚀 Custom Type Resolution Demo")
    print("=" * 50)

    # Register custom types
    register_custom_types()

    # Create a pipeline that uses custom types
    pipeline = Pipeline.from_steps(
        [
            Step.from_callable(process_user_data, name="process_user_data", updates_context=True),
            Step.from_callable(
                update_user_settings, name="update_user_settings", updates_context=True
            ),
        ]
    )

    # Create initial context with custom types
    initial_context = UserContext(
        current_user=UserProfile(name="John Doe", email="john@example.com", age=30),
        user_preferences=UserPreferences(favorite_colors=["blue", "green"], privacy_level="high"),
        complex_data=ComplexUserData(
            profile=UserProfile(name="John Doe", email="john@example.com", age=30),
            preferences=UserPreferences(favorite_colors=["blue", "green"], privacy_level="high"),
        ),
    )

    # Test data
    test_data = {"name": "John Doe", "theme": "dark", "notifications": False, "language": "es"}

    print("\n📋 Initial Context:")
    print(f"  User: {initial_context.current_user.name}")
    print(f"  Settings: {initial_context.user_settings}")
    print(f"  Preferences: {len(initial_context.user_preferences.favorite_colors)} colors")

    # Run the pipeline
    print("\n🔄 Running pipeline...")
    async for result in pipeline.run_async(
        test_data, initial_context_data=initial_context.model_dump()
    ):
        print(f"  Step: {result.step_name}")
        print(f"  Status: {result.status}")
        if result.error:
            print(f"  Error: {result.error}")

    print("\n✅ Demo completed successfully!")
    print("\n🎯 Key Benefits Demonstrated:")
    print("  • Custom types automatically resolved")
    print("  • No hardcoded type names needed")
    print("  • Secure frame stack traversal")
    print("  • High performance type resolution")
    print("  • Extensible for any user-defined types")


if __name__ == "__main__":
    asyncio.run(main())
