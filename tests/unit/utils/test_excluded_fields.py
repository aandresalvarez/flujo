from __future__ import annotations

import importlib
from typing import Set


def _reset_context_module():
    # Reload module to reset globals between tests
    import flujo.utils.context as ctx

    ctx._EXCLUDED_FIELDS_CACHE = None
    ctx._ENV_EXCLUDED_FIELDS_CACHE = None
    importlib.reload(ctx)
    return ctx


def _defaults() -> Set[str]:
    ctx = _reset_context_module()
    # Force default path by clearing env cache to empty string
    ctx._EXCLUDED_FIELDS_CACHE = None
    ctx._ENV_EXCLUDED_FIELDS_CACHE = ""
    return ctx.get_excluded_fields()


def test_excluded_fields_non_string_env_falls_back_to_defaults():
    ctx = _reset_context_module()
    ctx._EXCLUDED_FIELDS_CACHE = None
    # Simulate a non-string env value cached by the module
    ctx._ENV_EXCLUDED_FIELDS_CACHE = 123  # type: ignore[assignment]
    vals = ctx.get_excluded_fields()
    # Expect default set (non-empty; contains canonical defaults)
    assert "command_log" in vals
    assert "cache_timestamps" in vals
    assert "cache_keys" in vals


def test_excluded_fields_accepts_whitelisted_values_only():
    ctx = _reset_context_module()
    ctx._EXCLUDED_FIELDS_CACHE = None
    ctx._ENV_EXCLUDED_FIELDS_CACHE = "foo,cache_hits"
    vals = ctx.get_excluded_fields()
    assert vals == {"cache_hits"}


def test_excluded_fields_rejects_overlength_and_keeps_valid():
    ctx = _reset_context_module()
    ctx._EXCLUDED_FIELDS_CACHE = None
    long_name = "x" * 60
    ctx._ENV_EXCLUDED_FIELDS_CACHE = f"{long_name},run_id"
    vals = ctx.get_excluded_fields()
    assert "run_id" in vals
    assert long_name not in vals
    # Should not fall back to defaults because we had at least one valid entry
    assert vals == {"run_id"}


def test_excluded_fields_invalid_identifier_and_non_whitelisted_are_ignored():
    ctx = _reset_context_module()
    ctx._EXCLUDED_FIELDS_CACHE = None
    # invalid identifier and a valid identifier but not in whitelist
    ctx._ENV_EXCLUDED_FIELDS_CACHE = "bad-name,good_name"
    vals = ctx.get_excluded_fields()
    # Expect fallback defaults since no valid whitelisted entries
    assert "command_log" in vals and "cache_keys" in vals
