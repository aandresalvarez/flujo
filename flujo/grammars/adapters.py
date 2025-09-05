"""
Experimental grammar adapter stubs for structured decoding backends.

These are Phase 2 placeholders for Outlines/XGrammar integrations. They
intentionally return simple strings and do not affect execution until wired to
model backends. Kept here for future expansion and unit testing.
"""

from __future__ import annotations

from typing import Any


def compile_outlines_regex(schema: dict[str, Any]) -> str:
    """Return a naive regex placeholder from a JSON Schema root type.

    This is not a complete JSON Schema to regex compiler; it only serves as a
    stub for future Outlines integration.
    """
    t = schema.get("type")
    if t == "object":
        return r"\{[\s\S]*\}"
    if t == "array":
        return r"\[[\s\S]*\]"
    if t == "string":
        return r"\"[\s\S]*?\""
    if t == "integer":
        return r"-?\d+"
    if t == "number":
        return r"-?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?"
    if t == "boolean":
        return r"(?:true|false)"
    return r"[\s\S]+"


def compile_xgrammar(schema: dict[str, Any]) -> str:
    """Return a naive EBNF-like placeholder from a JSON Schema root type.

    Placeholder for a future XGrammar backend.
    """
    t = schema.get("type")
    if t == "object":
        return "object := '{' … '}' ;"  # illustrative only
    if t == "array":
        return "array := '[' … ']' ;"
    if t == "string":
        return "string := '" + '"' + "' … '" + '"' + "' ;"
    if t == "integer":
        return "integer := ['-'] DIGITS ;"
    if t == "number":
        return "number := ['-'] DIGITS ['.' DIGITS] ;"
    if t == "boolean":
        return "boolean := 'true' | 'false' ;"
    return "value := … ;"
