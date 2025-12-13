from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Callable, Any, Dict, List, Optional, Union

from flujo.type_definitions.common import JSONObject


# --- Fixer registry primitives ---

FixPreviewFn = Callable[[str, Any], int]


@dataclass(frozen=True)
class FixResult:
    """Result of applying a fixer to a single file.

    - ``applied``: whether any change was written
    - ``backup_path``: path to the backup file saved before modifications (if any)
    - ``changed``: number of discrete edits performed
    """

    applied: bool
    backup_path: Optional[str]
    changed: int


FixApplyFn = Callable[[str, Any, bool], FixResult]


class Fixer:
    def __init__(
        self, rule_id: str, preview: FixPreviewFn, apply: FixApplyFn, title: str = ""
    ) -> None:
        self.rule_id = rule_id
        self.preview = preview
        self.apply = apply
        self.title = title or rule_id


_REGISTRY: Dict[str, Fixer] = {}


def register_fixer(fixer: Fixer) -> None:
    _REGISTRY[fixer.rule_id.upper()] = fixer


def get_registered_fixers() -> Dict[str, Fixer]:
    return dict(_REGISTRY)


def _fix_vt1_in_text(yaml_text: str) -> Tuple[str, int]:
    """Rewrite previous_step.output misuses to use tojson filter.

    Strategy: simple textual replacement of 'previous_step.output' with
    'previous_step | tojson'. This is conservative and avoids parsing YAML.
    Returns the new text and number of replacements.
    """
    target = "previous_step.output"
    # Count non-overlapping occurrences
    count = yaml_text.count(target)
    if count == 0:
        return yaml_text, 0
    new_text = yaml_text.replace(target, "previous_step | tojson")
    return new_text, count


def count_findings(report: object, rule_id: str) -> int:
    try:
        errs = getattr(report, "errors", []) or []
        wrns = getattr(report, "warnings", []) or []
        return sum(1 for e in errs if getattr(e, "rule_id", None) == rule_id) + sum(
            1 for w in wrns if getattr(w, "rule_id", None) == rule_id
        )
    except Exception:
        return 0


# --- V-T1 Fixer implementation ---


def _vt1_preview(file_path: str, report: Any) -> int:
    try:
        text = Path(file_path).read_text(encoding="utf-8")
    except OSError:
        return 0
    # Use both report findings and text occurrences; take the min non-zero if both present
    in_text = text.count("previous_step.output")
    in_report = count_findings(report, "V-T1")
    if in_report > 0 and in_text > 0:
        return min(in_text, in_report)
    return in_text or in_report


def _vt1_apply(file_path: str, report: Any, assume_yes: bool) -> FixResult:
    p = Path(file_path)
    try:
        original = p.read_text(encoding="utf-8")
    except Exception:
        return FixResult(False, None, 0)
    new_text, replaced = _fix_vt1_in_text(original)
    if replaced <= 0 or new_text == original:
        return FixResult(False, None, 0)

    # Confirm if interactive and not assume_yes
    import sys

    if not assume_yes and getattr(sys.stdin, "isatty", lambda: False)():
        try:
            from typer import confirm as _confirm
        except ImportError:
            return FixResult(False, None, 0)
        if not _confirm(f"Apply {replaced} V-T1 fix(es) to {p.name}?", default=True):
            return FixResult(False, None, 0)

    backup_path = str(p) + ".bak"
    try:
        Path(backup_path).write_text(original, encoding="utf-8")
        p.write_text(new_text, encoding="utf-8")
    except OSError:
        return FixResult(False, None, 0)
    return FixResult(True, backup_path, replaced)


# Register V-T1
register_fixer(
    Fixer("V-T1", _vt1_preview, _vt1_apply, title="Rewrite previous_step.output → tojson")
)

# --- V-T3 Fixer: correct common filter typos ---

_VT3_MAP = {
    "to_json": "tojson",
    "to-json": "tojson",
    "toJson": "tojson",
    "lowercase": "lower",
    "uppercase": "upper",
    "len": "length",
    "joiner": "join",
    "jsonify": "tojson",
}


def _vt3_preview(file_path: str, report: Any) -> int:
    try:
        text = Path(file_path).read_text(encoding="utf-8")
    except OSError:
        return 0
    import re as _re

    count = 0
    for bad, _good in _VT3_MAP.items():
        pat = _re.compile(r"\|\s*" + _re.escape(bad) + r"\b")
        count += len(pat.findall(text))
    return count


def _apply_vt3_text(text: str) -> Tuple[str, int]:
    import re as _re

    changed = 0
    new_text = text
    for bad, good in _VT3_MAP.items():
        pat = _re.compile(r"(\|\s*)" + _re.escape(bad) + r"\b")

        def _sub(m: _re.Match[str], replacement: str = good) -> str:
            nonlocal changed
            changed += 1
            return m.group(1) + replacement

        new_text = pat.sub(_sub, new_text)
    return new_text, changed


def _vt3_apply(file_path: str, report: Any, assume_yes: bool) -> FixResult:
    p = Path(file_path)
    try:
        original = p.read_text(encoding="utf-8")
    except OSError:
        return FixResult(False, None, 0)
    new_text, changed = _apply_vt3_text(original)
    if changed <= 0 or new_text == original:
        return FixResult(False, None, 0)

    # Confirm prompt if interactive
    import sys

    if not assume_yes and getattr(sys.stdin, "isatty", lambda: False)():
        try:
            from typer import confirm as _confirm
        except ImportError:
            return FixResult(False, None, 0)
        if not _confirm(f"Apply {changed} V-T3 filter fix(es) to {p.name}?", default=True):
            return FixResult(False, None, 0)

    backup_path = str(p) + ".bak"
    try:
        Path(backup_path).write_text(original, encoding="utf-8")
        p.write_text(new_text, encoding="utf-8")
    except OSError:
        return FixResult(False, None, 0)
    return FixResult(True, backup_path, changed)


register_fixer(Fixer("V-T3", _vt3_preview, _vt3_apply, title="Fix common template filter typos"))

# --- V-C2 Fixer: avoid mapping directly to scratchpad root ---


def _vc2_preview(file_path: str, report: Any) -> int:
    try:
        text = Path(file_path).read_text(encoding="utf-8")
    except OSError:
        return 0
    import re as _re

    # Count both block and inline mapping occurrences
    pat_block = _re.compile(r"(^|\n)\s*parent:\s*[\"']?scratchpad[\"']?\s*(\n|$)")
    pat_inline = _re.compile(r"parent:\s*[\"']?scratchpad[\"']?(?=[,\s}])")
    return len(pat_block.findall(text)) + len(pat_inline.findall(text))


def _apply_vc2_text(text: str) -> Tuple[str, int]:
    import re as _re

    changed = 0
    new_text = text
    # Block style replacement
    pat_block = _re.compile(r"(^|\n)(\s*parent:\s*)([\"']?scratchpad[\"']?)(\s*(\n|$))")

    def _sub_block(m: _re.Match[str]) -> str:
        nonlocal changed
        changed += 1
        return f"{m.group(1)}{m.group(2)}import_artifacts.value{m.group(4)}"

    new_text = pat_block.sub(_sub_block, new_text)

    # Inline style replacement within braces
    pat_inline = _re.compile(r"(parent:\s*)([\"']?scratchpad[\"']?)(?=[,\s}])")

    def _sub_inline(m: _re.Match[str]) -> str:
        nonlocal changed
        changed += 1
        return f"{m.group(1)}import_artifacts.value"

    new_text = pat_inline.sub(_sub_inline, new_text)
    return new_text, changed


def _vc2_apply(file_path: str, report: Any, assume_yes: bool) -> FixResult:
    p = Path(file_path)
    try:
        original = p.read_text(encoding="utf-8")
    except OSError:
        return FixResult(False, None, 0)
    new_text, changed = _apply_vc2_text(original)
    if changed <= 0 or new_text == original:
        return FixResult(False, None, 0)

    # Confirm prompt if interactive
    import sys

    if not assume_yes and getattr(sys.stdin, "isatty", lambda: False)():
        try:
            from typer import confirm as _confirm
        except ImportError:
            return FixResult(False, None, 0)
        if not _confirm(f"Apply {changed} V-C2 parent mapping fix(es) to {p.name}?", default=True):
            return FixResult(False, None, 0)

    backup_path = str(p) + ".bak"
    try:
        Path(backup_path).write_text(original, encoding="utf-8")
        p.write_text(new_text, encoding="utf-8")
    except OSError:
        return FixResult(False, None, 0)
    return FixResult(True, backup_path, changed)


register_fixer(
    Fixer(
        "V-C2",
        _vc2_preview,
        _vc2_apply,
        title="Map to import_artifacts.<key> instead of removed root",
    )
)


# --- V-A8 Fixer: structured intent without JSON response mode ---


def _va8_preview(file_path: str, report: Any) -> int:
    # Prefer report counts when available
    n = count_findings(report, "V-A8")
    if n > 0:
        return n
    # Fallback: try lightweight YAML inspection
    try:
        import yaml as _yaml

        data = _yaml.safe_load(Path(file_path).read_text(encoding="utf-8"))
        steps = (data or {}).get("steps") or []
        cnt = 0
        for st in steps:
            try:
                proc = (st or {}).get("processing") or {}
                has_schema = bool(proc.get("schema")) or bool(
                    (st or {}).get("agent", {}).get("output_schema")
                )
                mode = (proc.get("structured_output") or "off").strip().lower()
                if has_schema and mode not in {"openai_json", "auto"}:
                    cnt += 1
            except Exception:
                continue
        return cnt
    except Exception:
        return 0


def _apply_va8_text(text: str) -> Tuple[str, int]:
    try:
        import yaml as _yaml
    except Exception:
        return text, 0
    try:
        data = _yaml.safe_load(text)
        changed = 0
        if not isinstance(data, dict):
            return text, 0
        steps = data.get("steps")
        if not isinstance(steps, list):
            return text, 0
        for st in steps:
            if not isinstance(st, dict):
                continue
            proc = st.get("processing")
            if not isinstance(proc, dict):
                # Only add when schema intent exists under processing or agent; create block if needed
                agent = st.get("agent") if isinstance(st.get("agent"), dict) else {}
                if isinstance(agent, dict) and agent.get("output_schema"):
                    st["processing"] = {"structured_output": "openai_json"}
                    changed += 1
                continue
            has_schema = bool(proc.get("schema"))
            mode = str(proc.get("structured_output", "")).strip().lower()
            if has_schema and mode not in {"openai_json", "auto"}:
                proc["structured_output"] = "openai_json"
                changed += 1
        if changed <= 0:
            return text, 0
        new_text = _yaml.safe_dump(data, sort_keys=False)
        return new_text, changed
    except Exception:
        return text, 0


def _va8_apply(file_path: str, _report: Any, assume_yes: bool) -> FixResult:
    p = Path(file_path)
    try:
        original = p.read_text(encoding="utf-8")
    except OSError:
        return FixResult(False, None, 0)
    new_text, changed = _apply_va8_text(original)
    if changed <= 0 or new_text == original:
        return FixResult(False, None, 0)

    import sys

    if not assume_yes and getattr(sys.stdin, "isatty", lambda: False)():
        try:
            from typer import confirm as _confirm
        except ImportError:
            return FixResult(False, None, 0)
        if not _confirm(f"Apply {changed} V-A8 structured_output fixes to {p.name}?", default=True):
            return FixResult(False, None, 0)

    backup_path = str(p) + ".bak"
    try:
        Path(backup_path).write_text(original, encoding="utf-8")
        p.write_text(new_text, encoding="utf-8")
    except OSError:
        return FixResult(False, None, 0)
    return FixResult(True, backup_path, changed)


register_fixer(
    Fixer(
        "V-A8",
        _va8_preview,
        _va8_apply,
        title="Enable processing.structured_output: openai_json when a schema is present",
    )
)


# --- Dry-run / patch preview ---
def build_fix_patch(
    path: str, report: Any, rules: Optional[List[str]] = None
) -> Tuple[str, JSONObject]:
    """Construct a unified diff preview of changes without writing files.

    Returns (patch_text, metrics)
    """
    from fnmatch import fnmatch
    import difflib

    p = Path(path)
    try:
        original = p.read_text(encoding="utf-8").splitlines(keepends=True)
    except Exception:
        return "", {"applied": {}, "total_applied": 0}

    def _allowed(rid: str) -> bool:
        if not rules:
            return True
        ridu = rid.upper()
        for pat in rules:
            try:
                if fnmatch(ridu, str(pat).upper()):
                    return True
            except Exception:
                continue
        return False

    text = "".join(original)
    total = 0
    per_rule: Dict[str, int] = {}
    # Apply in deterministic order
    if _allowed("V-T1"):
        text2, c = _fix_vt1_in_text(text)
        if c:
            per_rule["V-T1"] = per_rule.get("V-T1", 0) + c
            total += c
            text = text2
    if _allowed("V-T3"):
        text2, c = _apply_vt3_text(text)
        if c:
            per_rule["V-T3"] = per_rule.get("V-T3", 0) + c
            total += c
            text = text2
    if _allowed("V-C2"):
        text2, c = _apply_vc2_text(text)
        if c:
            per_rule["V-C2"] = per_rule.get("V-C2", 0) + c
            total += c
            text = text2
    if _allowed("V-A8"):
        text2, c = _apply_va8_text(text)
        if c:
            per_rule["V-A8"] = per_rule.get("V-A8", 0) + c
            total += c
            text = text2

    patched = text.splitlines(keepends=True)
    if total <= 0:
        return "", {"applied": {}, "total_applied": 0}
    diff = difflib.unified_diff(
        original,
        patched,
        fromfile=str(p) + ".orig",
        tofile=str(p),
        lineterm="",
    )
    return "\n".join(diff), {"applied": per_rule, "total_applied": total}


def plan_fixes(path: str, report: object, *, rules: Optional[List[str]] = None) -> list[JSONObject]:
    """Return a preview plan of fixable changes per rule.

    rules: optional allow-list of rule ids/globs (e.g., ["V-T1"]).
    """
    from fnmatch import fnmatch

    def _allowed(rid: str) -> bool:
        if not rules:
            return True
        ridu = rid.upper()
        for pat in rules:
            try:
                if fnmatch(ridu, str(pat).upper()):
                    return True
            except Exception:
                continue
        return False

    plan: list[JSONObject] = []
    for rid, fx in get_registered_fixers().items():
        if not _allowed(rid):
            continue
        try:
            count = fx.preview(path, report)
        except Exception:
            count = 0
        if count > 0:
            plan.append({"rule_id": rid, "count": count, "title": fx.title})
    return plan


def apply_fixes_to_file(
    path: str, report: object, *, assume_yes: bool = False, rules: Optional[List[str]] = None
) -> tuple[bool, Optional[str], JSONObject]:
    """Apply registered auto-fixes for the given file based on the report.

    Returns (applied_any, first_backup_path, metrics)
    metrics includes per-rule counts and total_applied.
    """
    metrics: JSONObject = {"applied": {}, "total_applied": 0}
    backup_first: Optional[str] = None
    applied_any = False

    # Execute fixers in deterministic rule-id order
    plan = plan_fixes(path, report, rules=rules)
    for item in plan:
        rid = item["rule_id"]
        fx = _REGISTRY.get(rid)
        if not fx:
            continue
        try:
            res: Union[FixResult, Tuple[bool, Optional[str], int]] = fx.apply(
                path, report, assume_yes
            )
        except Exception:
            res = FixResult(False, None, 0)

        if isinstance(res, tuple):
            ok, backup, changed = res
        else:
            ok, backup, changed = res.applied, res.backup_path, res.changed
        if ok and backup and not backup_first:
            backup_first = backup
        if changed > 0:
            applied_any = True
            # mypy: metrics['applied'] is a dict[str, Any]
            prev = metrics["applied"].get(rid, 0)
            try:
                prev_int = int(prev)
            except Exception:
                prev_int = 0
            metrics["applied"][rid] = prev_int + changed
            metrics["total_applied"] = int(metrics["total_applied"]) + changed
    return applied_any, backup_first, metrics
