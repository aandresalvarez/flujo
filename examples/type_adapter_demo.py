#!/usr/bin/env python3
"""
Demonstration of TypeAdapter support in flujo agent creation.

This example shows how to use Pydantic TypeAdapter instances with the
flujo agent factory, which automatically unwraps them to extract the
underlying type for proper schema generation and validation.
"""

import asyncio
from typing import List, Dict, Union
from pydantic import BaseModel, TypeAdapter
from flujo import make_agent_async


class User(BaseModel):
    """Example user model."""
    name: str
    id: int


class Product(BaseModel):
    """Example product model."""
    name: str
    price: float


async def demo_basic_type_adapter():
    """Demonstrate basic TypeAdapter usage with List[User]."""
    print("=== Basic TypeAdapter Demo ===")
    
    # Create a TypeAdapter for a list of users
    list_of_users_type = TypeAdapter(List[User])
    
    # Create an agent that returns a list of users
    agent = make_agent_async(
        model="openai:gpt-4o",
        system_prompt="Return a list of 3 example users with names and IDs.",
        output_type=list_of_users_type
    )
    
    print(f"Agent created successfully!")
    print(f"Target output type: {agent.target_output_type}")
    print(f"Expected type: {List[User]}")
    print(f"Types match: {agent.target_output_type == List[User]}")
    print()


async def demo_complex_nested_type_adapter():
    """Demonstrate complex nested TypeAdapter usage."""
    print("=== Complex Nested TypeAdapter Demo ===")
    
    # Create a TypeAdapter for a complex nested type
    complex_type = TypeAdapter(List[Dict[str, User]])
    
    # Create an agent that returns a complex nested structure
    agent = make_agent_async(
        model="openai:gpt-4o",
        system_prompt="Return a list of dictionaries mapping role names to user objects.",
        output_type=complex_type
    )
    
    print(f"Agent created successfully!")
    print(f"Target output type: {agent.target_output_type}")
    print(f"Expected type: {List[Dict[str, User]]}")
    print(f"Types match: {agent.target_output_type == List[Dict[str, User]]}")
    print()


async def demo_union_type_adapter():
    """Demonstrate TypeAdapter usage with Union types."""
    print("=== Union TypeAdapter Demo ===")
    
    # Create a TypeAdapter for a Union type
    union_type = TypeAdapter(Union[User, Product])
    
    # Create an agent that returns either a user or product
    agent = make_agent_async(
        model="openai:gpt-4o",
        system_prompt="Return either a user or a product object.",
        output_type=union_type
    )
    
    print(f"Agent created successfully!")
    print(f"Target output type: {agent.target_output_type}")
    print(f"Expected type: {Union[User, Product]}")
    print(f"Types match: {agent.target_output_type == Union[User, Product]}")
    print()


async def demo_regular_type_comparison():
    """Demonstrate that regular types work the same as before."""
    print("=== Regular Type Demo ===")
    
    # Create an agent with a regular type (not TypeAdapter)
    agent = make_agent_async(
        model="openai:gpt-4o",
        system_prompt="Return a single user object.",
        output_type=User
    )
    
    print(f"Agent created successfully!")
    print(f"Target output type: {agent.target_output_type}")
    print(f"Expected type: {User}")
    print(f"Types match: {agent.target_output_type == User}")
    print()


async def main():
    """Run all TypeAdapter demonstrations."""
    print("TypeAdapter Support in Flujo Agent Creation")
    print("=" * 50)
    print()
    
    await demo_basic_type_adapter()
    await demo_complex_nested_type_adapter()
    await demo_union_type_adapter()
    await demo_regular_type_comparison()
    
    print("All demonstrations completed successfully!")
    print("\nKey points:")
    print("- TypeAdapter instances are automatically unwrapped")
    print("- The underlying type is extracted and used for schema generation")
    print("- Regular types continue to work as before")
    print("- Complex nested types and Union types are supported")


if __name__ == "__main__":
    asyncio.run(main()) 