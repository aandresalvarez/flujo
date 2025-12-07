import asyncio

import pytest

from flujo.domain.models import Quota, UsageEstimate


@pytest.mark.asyncio
async def test_quota_reserve_race_condition() -> None:
    """Concurrent reservations must not overspend tokens."""

    quota = Quota(remaining_cost_usd=0.0, remaining_tokens=100)
    estimate = UsageEstimate(cost_usd=0.0, tokens=1)

    async def reserve_once() -> bool:
        return await asyncio.to_thread(quota.reserve, estimate)

    # Launch many concurrent reservations
    results = await asyncio.gather(*(reserve_once() for _ in range(100)))

    # All reservations should succeed exactly, with no overspend
    assert all(results)
    remaining_cost, remaining_tokens = quota.get_remaining()
    assert remaining_cost == 0.0
    assert remaining_tokens == 0
