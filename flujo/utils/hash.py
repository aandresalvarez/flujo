"""flujo.utils.hash

Safe, deterministic hashing helpers for cache key generation and other purposes.

This module deliberately avoids unsafe mechanisms such as pickle-based
serialization.  Instead, it constructs a stable SHA-256 digest by walking the
object graph and feeding canonical byte representations into the hasher.

Design goals
------------
1. **Security** – Never execute or evaluate arbitrary code.
2. **Determinism** – Equal objects → equal digests; order preserved for lists
   and tuples; keys sorted for mappings and sets.
3. **Robustness** – Gracefully handle circular references, deep nesting, and
   unhashable objects.
4. **Performance** – Stream data into the hasher incrementally to minimise
   intermediate allocations for large objects.
"""

from __future__ import annotations

import hashlib
from typing import Any, Set

__all__ = ["stable_digest", "hash_bytes"]

_MAX_DEPTH = 10  # Prevent runaway recursion


def _update_with_separator(hasher: hashlib._Hash, sep: str) -> None:
    """Feed a separator string into the hasher (as UTF-8)."""
    hasher.update(sep.encode())


def _hash_obj(obj: Any, hasher: hashlib._Hash, visited: Set[int], depth: int) -> None:
    """Recursively update *hasher* with a deterministic representation of *obj*."""
    if depth > _MAX_DEPTH:
        hasher.update(b"<max_depth>")
        return

    obj_id = id(obj)
    if obj_id in visited:
        hasher.update(b"<circular>")
        return

    # Track visited to avoid infinite recursion
    visited.add(obj_id)

    try:
        if obj is None:
            hasher.update(b"null")
        elif isinstance(obj, (bool, int, float)):
            hasher.update(repr(obj).encode())
        elif isinstance(obj, str):
            hasher.update(obj.encode())
        elif isinstance(obj, (list, tuple)):
            _update_with_separator(hasher, "[")
            for item in obj:
                _hash_obj(item, hasher, visited, depth + 1)
                _update_with_separator(hasher, ",")
            _update_with_separator(hasher, "]")
        elif isinstance(obj, dict):
            # Sort keys for deterministic ordering
            _update_with_separator(hasher, "{")
            for key in sorted(obj.keys(), key=str):
                _hash_obj(key, hasher, visited, depth + 1)
                _update_with_separator(hasher, ":")
                _hash_obj(obj[key], hasher, visited, depth + 1)
                _update_with_separator(hasher, ",")
            _update_with_separator(hasher, "}")
        elif isinstance(obj, set):
            _update_with_separator(hasher, "{")
            for item in sorted(obj, key=str):
                _hash_obj(item, hasher, visited, depth + 1)
                _update_with_separator(hasher, ",")
            _update_with_separator(hasher, "}")
        else:
            # Fallback to type name + string representation
            hasher.update(f"<{type(obj).__name__}:{str(obj)}>".encode())
    finally:
        visited.discard(obj_id)


def hash_bytes(obj: Any) -> bytes:
    """Return SHA-256 digest *bytes* for *obj* using stable hashing."""
    hasher = hashlib.sha256()
    _hash_obj(obj, hasher, visited=set(), depth=0)
    return hasher.digest()


def stable_digest(obj: Any) -> str:
    """Return SHA-256 hexadecimal digest string for *obj*."""
    return hashlib.sha256(hash_bytes(obj)).hexdigest()
