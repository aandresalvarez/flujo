from __future__ import annotations

from typing import Any


async def echo(x: Any) -> Any:
    return x


def make_echo(**_params: Any) -> Any:
    # Factory returns the async callable used by the Step agent wrapper
    return echo
