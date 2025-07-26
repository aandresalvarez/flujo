#!/usr/bin/env python3
"""
Demonstration of the new image generation client with cost tracking.

This example shows how to use the get_image_client factory to generate images
with automatic cost tracking that integrates with Flujo's cost management system.
"""

import asyncio
from flujo.images import get_image_client


async def demo_image_generation():
    """Demonstrate image generation with cost tracking."""

    print("🚀 Image Generation Client Demo")
    print("=" * 50)

    # Get the image client (requires openai:dall-e-3 pricing in flujo.toml)
    try:
        client = get_image_client("openai:dall-e-3")
        print("✅ Successfully created image client")
    except Exception as e:
        print(f"❌ Failed to create image client: {e}")
        print("💡 Make sure you have DALL-E 3 pricing configured in flujo.toml:")
        print("""
[cost.providers.openai.dall-e-3]
price_per_image_standard_1024x1024 = 0.040
price_per_image_standard_1024x1792 = 0.080
price_per_image_hd_1024x1024 = 0.080
        """)
        return

    # Generate an image
    prompt = "A serene landscape with mountains and a lake at sunset"
    print(f"\n🎨 Generating image: '{prompt}'")

    try:
        result = client.generate(
            prompt=prompt,
            size="1024x1024",
            quality="standard"
        )

        print("✅ Image generated successfully!")
        print(f"💰 Cost: ${result.cost_usd:.4f}")
        print(f"🔢 Token count: {result.token_counts}")
        print(f"🖼️  Number of images: {len(result.image_urls)}")

        for i, url in enumerate(result.image_urls, 1):
            print(f"   Image {i}: {url}")

    except Exception as e:
        print(f"❌ Failed to generate image: {e}")


async def demo_cost_integration():
    """Demonstrate how image costs integrate with other cost tracking."""

    print("\n" + "=" * 50)
    print("💰 Cost Integration Demo")
    print("=" * 50)

    # Simulate a pipeline with both chat and image steps
    class DummyChatResult:
        cost_usd = 0.02
        token_counts = 1000

    class DummyImageResult:
        cost_usd = 0.04
        token_counts = 0

    chat_result = DummyChatResult()
    image_result = DummyImageResult()

    total_cost = chat_result.cost_usd + image_result.cost_usd
    total_tokens = chat_result.token_counts + image_result.token_counts

    print(f"💬 Chat step cost: ${chat_result.cost_usd:.4f} ({chat_result.token_counts} tokens)")
    print(f"🖼️  Image step cost: ${image_result.cost_usd:.4f} ({image_result.token_counts} tokens)")
    print(f"📊 Total cost: ${total_cost:.4f}")
    print(f"📊 Total tokens: {total_tokens}")

    # Demonstrate usage limit enforcement
    usage_limit = 0.05
    if total_cost > usage_limit:
        print(f"⚠️  Usage limit exceeded! (${usage_limit:.4f} limit, ${total_cost:.4f} actual)")
    else:
        print(f"✅ Within usage limit (${usage_limit:.4f})")


if __name__ == "__main__":
    asyncio.run(demo_image_generation())
    asyncio.run(demo_cost_integration())
