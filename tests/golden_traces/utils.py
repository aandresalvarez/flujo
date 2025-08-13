from __future__ import annotations

from typing import Any, Dict, List


def _attributes_keys(attrs: Dict[str, Any] | None) -> List[str]:
    if not attrs:
        return []
    return sorted(list(attrs.keys()))


def span_to_contract_dict(span: Any) -> Dict[str, Any]:
    """Convert a TraceManager Span to a contract-comparable dict.

    Keeps only stable fields: name, attribute keys, event names+attribute keys, and children.
    """
    node: Dict[str, Any] = {
        "name": getattr(span, "name", "<unknown>"),
        "attributes": _attributes_keys(getattr(span, "attributes", {})),
        "events": [],
        "children": [],
    }
    events = getattr(span, "events", []) or []
    for ev in events:
        node["events"].append(
            {
                "name": ev.get("name"),
                "attributes": _attributes_keys(ev.get("attributes", {})),
            }
        )
    for child in getattr(span, "children", []) or []:
        node["children"].append(span_to_contract_dict(child))
    # Ensure deterministic ordering by child name
    node["events"].sort(key=lambda e: (e.get("name") or ""))
    node["children"].sort(key=lambda c: (c.get("name") or ""))
    return node


def normalize_contract_tree(tree: Dict[str, Any]) -> Dict[str, Any]:
    """Sort nested lists to ensure order-insensitive comparison."""
    tree = {
        "name": tree.get("name"),
        "attributes": sorted(tree.get("attributes", [])),
        "events": sorted(
            (
                {"name": e.get("name"), "attributes": sorted(e.get("attributes", []))}
                for e in tree.get("events", [])
            ),
            key=lambda e: (e.get("name") or ""),
        ),
        "children": [],
    }
    children = []
    for ch in tree.get("children", []):
        children.append(normalize_contract_tree(ch))
    tree["children"] = sorted(children, key=lambda c: (c.get("name") or ""))
    return tree


def trees_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return normalize_contract_tree(a) == normalize_contract_tree(b)
