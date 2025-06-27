import inspect
import warnings
from typing import Any, Type

_warned_locations: set[tuple[str, int]] = set()


def warn_once(
    message: str, category: Type[Warning] = DeprecationWarning, stacklevel: int = 2
) -> None:
    frame = inspect.stack()[stacklevel]
    loc = (frame.filename, frame.lineno)
    if loc in _warned_locations:
        return
    _warned_locations.add(loc)
    warnings.warn(message, category, stacklevel=stacklevel)
