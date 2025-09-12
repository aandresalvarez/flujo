from __future__ import annotations

"""
Minimal JSON normalization helpers for AROS Phase 2.

These functions are intentionally conservative: they strip common markdown
fences, attempt a best-effort extraction of the largest JSON object/array,
and parse with stdlib json, falling back to the original text on failure.
"""

from typing import Any  # noqa: E402


def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```json") and t.endswith("```"):
        t = t[len("```json") :].strip()
        if t.endswith("```"):
            t = t[: -len("```")].strip()
        return t
    if t.startswith("```") and t.endswith("```"):
        t = t[len("```") :].strip()
        if t.endswith("```"):
            t = t[: -len("```")].strip()
        return t
    return text


def extract_likely_json_region(text: str) -> str:
    """Extract the largest balanced {...} or [...] region; fall back to input."""
    s = text
    start_obj = s.find("{")
    start_arr = s.find("[")
    spans = []
    for start, open_ch, close_ch in ((start_obj, "{", "}"), (start_arr, "[", "]")):
        if start == -1:
            continue
        depth = 0
        end = -1
        for i in range(start, len(s)):
            ch = s[i]
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end != -1:
            spans.append((start, end))
    if not spans:
        return text
    # Choose the longest
    spans.sort(key=lambda p: p[1] - p[0], reverse=True)
    start, end = spans[0]
    return s[start:end]


def normalize_to_json_obj(text: str) -> Any:
    """Return parsed JSON object when possible; else raise or return original text."""
    import json

    if not isinstance(text, str):
        return text
    t = strip_code_fences(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    region = extract_likely_json_region(t)
    if region != t:
        try:
            return json.loads(region)
        except Exception:
            pass
    # Trailing commas quick fix
    try:
        fixed = t.replace(",\n}\n", "\n}\n").replace(",\n]\n", "\n]\n")
        return json.loads(fixed)
    except Exception:
        pass
    return text
