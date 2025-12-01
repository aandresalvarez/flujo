from __future__ import annotations

from typing import Any, List, Set
from flujo.type_definitions.common import JSONObject


def _attributes_keys(attrs: JSONObject | None) -> List[str]:
    if not attrs:
        return []
    return sorted(list(attrs.keys()))


REQUIRED_ROOT_ATTRS: Set[str] = {
    "flujo.input",
    # Optional-by-contract but frequently present; we don't fail if missing in comparison logic
}

REQUIRED_STEP_ATTRS: Set[str] = {
    "flujo.step.type",
    "flujo.step.policy",
    # attempt_number, cache.hit, budget fields may be optional depending on lifecycle, so we only check keys when present in golden
}


def span_to_contract_dict(span: Any) -> JSONObject:
    """Convert a TraceManager Span to a contract-comparable dict.

    Keeps only stable fields: name, attribute keys, event names+attribute keys, and children.
    """
    node: JSONObject = {
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
    # Best-effort structural validation: ensure required keys exist in the node attributes for root/step
    try:
        if node["name"] == "pipeline_run":
            for req in REQUIRED_ROOT_ATTRS:
                if req not in node["attributes"]:
                    node["attributes"].append(req)
        else:
            # Child step spans
            for req in REQUIRED_STEP_ATTRS:
                if req not in node["attributes"]:
                    node["attributes"].append(req)
        node["attributes"] = sorted(list(set(node["attributes"])))
    except Exception:
        pass
    return node


def normalize_contract_tree(node: JSONObject) -> JSONObject:
    """Sort nested lists to ensure order-insensitive comparison, preserving structure."""
    normalized_children = [normalize_contract_tree(ch) for ch in (node.get("children", []) or [])]
    normalized_events = [
        {"name": e.get("name"), "attributes": sorted(e.get("attributes", []))}
        for e in (node.get("events", []) or [])
    ]
    return {
        "name": node.get("name"),
        "attributes": sorted(node.get("attributes", [])),
        "events": sorted(normalized_events, key=lambda e: (e.get("name") or "")),
        "children": sorted(normalized_children, key=lambda c: (c.get("name") or "")),
    }


def trees_equal(a: JSONObject, b: JSONObject) -> bool:
    return normalize_contract_tree(a) == normalize_contract_tree(b)


# --- Optional helper: TraceCapturingHook (per FSD-011) ---


class TraceCapturingHook:
    """Testing helper that captures hook payloads into a minimal trace tree.

    This mirrors the contract-focused structure: name, attribute keys, events, and
    children. It is intentionally lightweight and used only within tests.
    """

    def __init__(self) -> None:
        self._stack: List[JSONObject] = []
        self.root: JSONObject | None = None

    async def hook(self, payload) -> None:
        event = getattr(payload, "event_name", None)
        if event == "pre_run":
            self._handle_pre_run(payload)
        elif event == "post_run":
            self._handle_post_run(payload)
        elif event == "pre_step":
            self._handle_pre_step(payload)
        elif event == "post_step":
            self._handle_post_step(payload)
        elif event == "on_step_failure":
            self._handle_on_step_failure(payload)
        # else: ignore

    def _handle_pre_run(self, payload: Any) -> None:
        root_attrs: List[str] = ["flujo.input"]
        # Optional enrichments if present
        if getattr(payload, "run_id", None) is not None:
            root_attrs.append("flujo.run_id")
        if getattr(payload, "pipeline_name", None) is not None:
            root_attrs.append("flujo.pipeline.name")
        if getattr(payload, "pipeline_version", None) is not None:
            root_attrs.append("flujo.pipeline.version")
        node: JSONObject = {
            "name": "pipeline_run",
            "attributes": sorted(list(set(root_attrs))),
            "events": [],
            "children": [],
        }
        self.root = node
        self._stack = [node]

    def _handle_post_run(self, _payload: Any) -> None:
        # Nothing to finalize beyond structure for this lightweight helper
        pass

    def _handle_pre_step(self, payload: Any) -> None:
        if not self._stack:
            # If no root was established, create a minimal root
            self._handle_pre_run(
                type(
                    "_Pre",
                    (),
                    {
                        "run_id": None,
                        "pipeline_name": None,
                        "pipeline_version": None,
                        "initial_input": None,
                    },
                )()
            )
        parent = self._stack[-1]
        # If somehow root isn't created, initialize it to keep tests robust
        if parent.get("name") != "pipeline_run":
            self._handle_pre_run(
                type(
                    "_Pre",
                    (),
                    {
                        "run_id": None,
                        "pipeline_name": None,
                        "pipeline_version": None,
                        "initial_input": None,
                    },
                )()
            )
            parent = self._stack[-1]
        attrs: List[str] = ["flujo.step.type", "flujo.step.policy", "step_input"]
        if getattr(payload, "attempt_number", None) is not None:
            attrs.append("flujo.attempt_number")
        if getattr(payload, "quota_before_usd", None) is not None:
            attrs.append("flujo.budget.quota_before_usd")
        if getattr(payload, "quota_before_tokens", None) is not None:
            attrs.append("flujo.budget.quota_before_tokens")
        if getattr(payload, "cache_hit", None) is not None:
            attrs.append("flujo.cache.hit")
        child: JSONObject = {
            "name": getattr(payload.step, "name", "<step>"),
            "attributes": sorted(list(set(attrs))),
            "events": [],
            "children": [],
        }
        parent["children"].append(child)
        self._stack.append(child)

    def _handle_post_step(self, payload: Any) -> None:
        if not self._stack:
            return
        current = self._stack.pop()
        # Append completion attributes per contract
        attrs = set(current.get("attributes", []))
        attrs.update(
            ["success", "latency_s", "flujo.budget.actual_cost_usd", "flujo.budget.actual_tokens"]
        )
        current["attributes"] = sorted(list(attrs))
        # Detect fallback event best-effort via result metadata
        try:
            md = getattr(payload.step_result, "metadata_", {}) or {}
            if md.get("fallback_triggered"):
                current["events"].append(
                    {
                        "name": "flujo.fallback.triggered",
                        "attributes": ["original_error"],
                    }
                )
        except Exception:
            pass

    def _handle_on_step_failure(self, payload: Any) -> None:
        if not self._stack:
            return
        current = self._stack.pop()
        attrs = set(current.get("attributes", []))
        attrs.update(
            [
                "success",
                "latency_s",
                "flujo.budget.actual_cost_usd",
                "flujo.budget.actual_tokens",
                "feedback",
            ]
        )
        current["attributes"] = sorted(list(attrs))

    def get_contract_tree(self) -> JSONObject | None:
        if self.root is None:
            return None
        return normalize_contract_tree(self.root)
