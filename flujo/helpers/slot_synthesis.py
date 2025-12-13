from __future__ import annotations

import re
from typing import Any, Optional, Tuple, List
from flujo.domain.models import PipelineContext
from flujo.type_definitions.common import JSONObject


def _extract_years(text: str) -> Tuple[Optional[int], Optional[int]]:
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text)]
    if not years:
        return None, None
    if len(years) == 1:
        return years[0], years[0]
    return min(years), max(years)


def _extract_age_range(text: str) -> Tuple[Optional[int], Optional[int]]:
    # Common patterns: "20 to 30", "between 20 and 30", "20-30"
    m = re.search(r"\b(\d{1,3})\s*(?:to|\-|–|—|and)\s*(\d{1,3})\b", text)
    if m:
        a1, a2 = int(m.group(1)), int(m.group(2))
        return (min(a1, a2), max(a1, a2))
    # Fallback: capture single age when preceded by age/aged
    m2 = re.search(r"\b(?:age|aged)\s*(\d{1,3})\b", text)
    if m2:
        a = int(m2.group(1))
        return a, a
    return None, None


def _extract_sex(text: str) -> Optional[str]:
    low = text.lower()
    if any(w in low for w in ["male", "men", "man"]):
        return "male"
    if any(w in low for w in ["female", "women", "woman"]):
        return "female"
    return None


def _extract_metric(text: str) -> Optional[str]:
    low = text.lower()
    if "count" in low or "counts" in low:
        return "count"
    if "rate" in low:
        return "rate"
    return None


def _extract_grouping(text: str) -> List[str]:
    low = text.lower()
    if low.strip() in {"no", "none", "n/a", "na"}:
        return []
    dims = []
    for key in [
        "age",
        "sex",
        "gender",
        "location",
        "state",
        "region",
        "zip",
        "city",
        "country",
    ]:
        if key in low:
            dims.append("sex" if key == "gender" else key)
    # Heuristic: if user writes "by X and Y"
    by_parts = re.findall(r"by\s+([a-z,\s]+)", low)
    for part in by_parts:
        for token in re.split(r"[,\s]+", part):
            token = token.strip()
            if token in {
                "age",
                "sex",
                "gender",
                "location",
                "state",
                "region",
                "zip",
                "city",
                "country",
            }:
                dims.append("sex" if token == "gender" else token)
    # Deduplicate preserving order
    seen = set()
    out: List[str] = []
    for d in dims:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _extract_filters(text: str) -> Optional[str]:
    low = text.lower().strip()
    if low in {"no", "none", "n/a", "na", "nothing"}:
        return None
    return text.strip() or None


async def synthesize_slots(
    _: Any = None, *, context: Optional[PipelineContext] = None, **kwargs: Any
) -> JSONObject:
    """Synthesize structured clarification slots from HITL transcript.

    - Reads context.hitl_history (list of {message_to_human, human_response}).
    - Assigns replies to slots based on assistant question text.
    - Produces a consolidated slots dict under context.hitl_data.slots when used with updates_context: true.

    Returns a dict suitable for updates_context merging:
      { "hitl_data": { "slots": { ... }, "slots_filled": [...], "slots_missing": [...], "slots_text_summary": "..." } }
    """
    slots: JSONObject = {
        "metric": None,
        "cohort": {"sex": None, "age_min": None, "age_max": None},
        "time_window": {"start_year": None, "end_year": None},
        "grouping": [],
        "filters": None,
    }
    if context is None:
        return {
            "hitl_data": {"slots": slots, "slots_filled": [], "slots_missing": list(slots.keys())}
        }

    history = getattr(context, "hitl_history", None) or []
    for turn in history:
        try:
            q = str(getattr(turn, "message_to_human", "") or "")
            a = str(getattr(turn, "human_response", "") or "")
        except Exception:
            try:
                q = str(turn.get("message_to_human", ""))
                a = str(turn.get("human_response", ""))
            except Exception:
                q, a = "", ""
        ql = q.lower()
        if not a.strip():
            continue
        # Metric
        if "metric" in ql:
            m = _extract_metric(a)
            slots["metric"] = m or (a.strip())
            continue
        # Cohort/Population
        if ("cohort" in ql) or ("population" in ql):
            sex = _extract_sex(a)
            age_min, age_max = _extract_age_range(a)
            if sex:
                slots["cohort"]["sex"] = sex
            if age_min is not None:
                slots["cohort"]["age_min"] = age_min
            if age_max is not None:
                slots["cohort"]["age_max"] = age_max
            # Keep raw cohort text if nothing was parsed
            if not sex and age_min is None and age_max is None:
                slots["cohort"]["raw"] = a.strip()
            continue
        # Time window
        if (
            ("time window" in ql)
            or ("time-window" in ql)
            or ("time" in ql and ("year" in ql or "date" in ql))
        ):
            y1, y2 = _extract_years(a)
            if y1 is not None:
                slots["time_window"]["start_year"] = y1
            if y2 is not None:
                slots["time_window"]["end_year"] = y2
            if y1 is None and y2 is None:
                slots["time_window"]["raw"] = a.strip()
            continue
        # Grouping/Dimensions
        if ("group" in ql) or ("dimension" in ql):
            dims = _extract_grouping(a)
            slots["grouping"] = dims
            continue
        # Filters/Exclusions
        if ("filter" in ql) or ("exclusion" in ql):
            filt = _extract_filters(a)
            slots["filters"] = filt
            continue

    # Compute filled/missing
    filled: List[str] = []
    missing: List[str] = []
    for key in ["metric", "cohort", "time_window", "grouping", "filters"]:
        val = slots.get(key)
        is_filled = False
        if isinstance(val, dict):
            is_filled = any(v is not None and v != [] for v in val.values())
        elif isinstance(val, list):
            is_filled = len(val) > 0
        else:
            is_filled = val is not None and str(val).strip() != ""
        (filled if is_filled else missing).append(key)

    # Text summary
    summary_parts: List[str] = []
    if slots["metric"]:
        summary_parts.append(f"metric={slots['metric']}")
    co = slots["cohort"]
    co_parts: List[str] = []
    if co.get("sex"):
        co_parts.append(f"sex={co['sex']}")
    if co.get("age_min") is not None and co.get("age_max") is not None:
        co_parts.append(f"age={co['age_min']}-{co['age_max']}")
    if co.get("raw"):
        co_parts.append(f"raw={co['raw']}")
    if co_parts:
        summary_parts.append("cohort=" + ",".join(co_parts))
    tw = slots["time_window"]
    if tw.get("start_year") is not None and tw.get("end_year") is not None:
        summary_parts.append(f"time_window={tw['start_year']}-{tw['end_year']}")
    if tw.get("raw"):
        summary_parts.append(f"time_window_raw={tw['raw']}")
    if slots["grouping"]:
        summary_parts.append("grouping=" + ",".join(slots["grouping"]))
    if slots["filters"]:
        summary_parts.append(f"filters={slots['filters']}")
    slots_text_summary = "; ".join(summary_parts)

    return {
        "hitl_data": {
            "slots": slots,
            "slots_filled": filled,
            "slots_missing": missing,
            "slots_text_summary": slots_text_summary,
        }
    }
