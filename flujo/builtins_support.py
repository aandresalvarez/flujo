from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel as PydanticBaseModel

from flujo.domain.models import BaseModel as DomainBaseModel
from flujo.type_definitions.common import JSONObject

_ruamel_yaml: Any = None
try:
    import ruamel.yaml as _ruamel_yaml
except ImportError:
    pass

_pyfiglet: Any = None
try:  # pragma: no cover - optional dependency
    import pyfiglet as _pyfiglet
except Exception:
    pass


async def passthrough(
    x: JSONObject | str | list[JSONObject] | None,
) -> JSONObject | str | list[JSONObject] | None:
    """Return the input unchanged (identity)."""
    return x


async def repair_yaml_ruamel(yaml_text: str) -> JSONObject:
    """Conservatively attempt to repair malformed pipeline YAML text.

    Strategy:
    - Heuristic fix: if the line after 'steps:' contains an unmatched '[', rewrite as 'steps: []'.
    - If ruamel.yaml is available, perform a round-trip load/dump to normalize formatting while
      preserving quotes and structure. Ensure top-level keys like version exist.
    - Always return a mapping containing both 'generated_yaml' and 'yaml_text'.
    """
    # Normalize input
    text: str = yaml_text or ""

    # Heuristic patch for common malformed inline list after 'steps:'
    try:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("steps:"):
                tail = line.split("steps:", 1)[1]
                if "[" in tail and "]" not in tail:
                    # Preserve indentation when replacing the line
                    indent = line[: len(line) - len(line.lstrip(" "))]
                    lines[i] = f"{indent}steps: []"
                    text = "\n".join(lines)
                    break
    except Exception:
        # Best-effort heuristic; ignore failures and continue with original text
        pass

    # Round-trip parse and dump if ruamel is available to normalize YAML
    if _ruamel_yaml is not None:
        try:
            yaml = _ruamel_yaml.YAML()
            yaml.preserve_quotes = True
            data = yaml.load(text)
            # Ensure minimal keys and sane defaults
            if isinstance(data, dict):
                if "version" not in data:
                    data["version"] = "0.1"
                # Add a default pipeline name when missing to satisfy validators
                if "name" not in data:
                    data["name"] = "generated_pipeline"
                if "steps" in data and data["steps"] is None:
                    data["steps"] = []
            from io import StringIO

            buf = StringIO()
            yaml.dump(data, buf)
            fixed = buf.getvalue()
            return {"generated_yaml": fixed, "yaml_text": fixed}
        except Exception:
            # Fall through to return (possibly heuristically patched) text
            pass

    # Fallback: return text as-is (possibly after heuristic correction)
    return {"generated_yaml": text, "yaml_text": text}


# --- Adapter: return YAML in CLI-expected format ---
async def return_yaml_for_cli(yaml_text: str | JSONObject) -> Dict[str, str]:
    """Return YAML in the format that the CLI expects to find, extracting it if necessary.

    Defensive against LLMs that occasionally prepend prose before the YAML. If any
    non-YAML text precedes the first YAML key, trim everything before the first line
    starting with 'version:' or 'name:'.
    """
    import re

    # Handle dictionary input (from extract_yaml_text)
    if isinstance(yaml_text, dict):
        yaml_string = (
            yaml_text.get("yaml_text") or yaml_text.get("generated_yaml") or str(yaml_text)
        )
    else:
        yaml_string = str(yaml_text)

    # Defensive extraction: find the start of the YAML content
    try:
        match = re.search(r"^(version:|name:)", yaml_string, re.MULTILINE)
        if match:
            yaml_string = yaml_string[match.start() :]
    except Exception:
        # Best-effort; keep original string on regex failure
        pass

    return {"generated_yaml": yaml_string, "yaml_text": yaml_string}


# --- Welcome agent for new users ---
async def welcome_agent(name: str = "Developer") -> str:
    """
    Return a welcome message, optionally with ASCII art when pyfiglet is available.

    Gracefully degrades to plain text when pyfiglet (optional) is not installed
    or if the configured font is unavailable.
    """
    welcome_header = f"Welcome, {name}!"

    flujo_art = ""
    if _pyfiglet:
        try:
            fig = _pyfiglet.Figlet(font="slant")
            flujo_art = fig.renderText("Flujo")
        except Exception:
            # Fallback to plain text if font is missing or another error occurs
            flujo_art = "F L U J O\n"

    welcome_body = (
        "\nYou have successfully run your first pipeline!\n\n"
        "This is a simple workflow defined in `pipeline.yaml`.\n"
        "You can edit it or create a new one from scratch by running:\n\n"
        '  flujo create --goal "Your new workflow goal"\n\n'
        "Happy building!\n"
    )

    return f"{welcome_header}\n\n{flujo_art}{welcome_body}"


# --- Context manipulation helpers (Task 2.3) ---


async def context_set(
    path: str,
    value: Any,
    *,
    context: Optional[DomainBaseModel] = None,
) -> JSONObject:
    """Set a context field at the specified dot-separated path.

    This is a built-in skill that provides type-safe context manipulation,
    reducing boilerplate and preventing `Any` type usage.

    Args:
        path: Dot-separated path to the field (e.g., "call_count" or "import_artifacts.counter")
        value: Value to set at the path
        context: Pipeline context (injected automatically by Flujo)

    Returns:
        Dict with path and value for confirmation

    Example:
        ```yaml
        - kind: step
          name: init_counter
          agent: { id: "flujo.builtins.context_set" }
          input: { path: "call_count", value: 0 }
        ```
    """
    from flujo.utils.context import set_nested_context_field

    success = False

    if context is not None:
        try:
            set_nested_context_field(context, path, value)
            success = True
        except Exception as e:
            # Log warning but don't fail the step
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to set context path '{path}': {e}")

    return {"path": path, "value": value, "success": success}


async def context_merge(
    path: str,
    value: JSONObject,
    *,
    context: Optional[DomainBaseModel] = None,
) -> JSONObject:
    """Merge a dictionary into the context at the specified path.

    This is useful for updating nested context objects with multiple fields at once.

    Args:
        path: Dot-separated path to merge into (e.g., "hitl_data" or "import_artifacts.extras")
        value: Dictionary to merge at the path
        context: Pipeline context (injected automatically by Flujo)

    Returns:
        Dict with path, merged keys, and success status

    Example:
        ```yaml
        - kind: step
          name: update_settings
          agent: { id: "flujo.builtins.context_merge" }
          input:
            path: "import_artifacts.extras"
            value: { theme: "dark", notifications: true }
        ```
    """
    from flujo.utils.context import set_nested_context_field

    merged_keys: List[str] = []
    if context is not None and isinstance(value, dict):
        try:
            # Get the target object at the path
            parts = path.split(".")
            target = context
            for part in parts:
                try:
                    target = getattr(target, part)
                except AttributeError:
                    if isinstance(target, dict) and part in target:
                        target = target[part]
                    else:
                        # Path doesn't exist, create it first
                        set_nested_context_field(context, path, {})
                        target = context
                        for p in parts:
                            if hasattr(target, p):
                                target = getattr(target, p)
                            elif isinstance(target, dict):
                                target = target[p]
                            else:
                                # Can't traverse further
                                break

            # Merge the dictionary
            if isinstance(target, dict):
                target.update(value)
                merged_keys = list(value.keys())
            else:
                # If target is not a dict, try to set individual attributes
                for key, val in value.items():
                    try:
                        setattr(target, key, val)
                        merged_keys.append(key)
                    except (AttributeError, TypeError):
                        pass

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to merge into context path '{path}': {e}")

    return {"path": path, "merged_keys": merged_keys, "success": len(merged_keys) > 0}


async def context_get(
    path: str,
    default: Any = None,
    *,
    context: Optional[DomainBaseModel] = None,
) -> Any:
    """Get a value from the context at the specified dot-separated path.

    Args:
        path: Dot-separated path to the field (e.g., "call_count" or "import_artifacts.counter")
        default: Default value if path doesn't exist
        context: Pipeline context (injected automatically by Flujo)

    Returns:
        The value at the path, or default if not found

    Example:
        ```yaml
        - kind: step
          name: get_counter
          agent: { id: "flujo.builtins.context_get" }
          input: { path: "call_count", default: 0 }
        ```
    """
    if context is None:
        return default

    try:
        parts = path.split(".")
        target = context
        for part in parts:
            try:
                # Try attribute access first
                target = getattr(target, part)
            except AttributeError:
                # Try dict access
                if isinstance(target, dict) and part in target:
                    target = target[part]
                else:
                    return default
        return target
    except Exception:
        return default


async def extract_decomposed_steps(
    decomposition: PydanticBaseModel | JSONObject,
    *,
    output_key: str = "prepared_steps_for_mapping",
) -> JSONObject:
    """Extract a list of step dicts from a decomposer output structure."""
    steps: list[JSONObject] = []
    try:
        if isinstance(decomposition, PydanticBaseModel):
            raw = decomposition.model_dump()
            if isinstance(raw, dict):
                cand = raw.get("steps")
                if isinstance(cand, list):
                    steps = [x for x in cand if isinstance(x, dict)]
        elif isinstance(decomposition, dict):
            cand = decomposition.get("steps")
            if isinstance(cand, list):
                steps = [x for x in cand if isinstance(x, dict)]
        else:
            cand = getattr(decomposition, "steps", None)
            if isinstance(cand, list):
                steps = [x for x in cand if isinstance(x, dict)]
    except Exception:
        steps = []
    return {output_key: steps}


async def extract_yaml_text(
    writer_output: PydanticBaseModel | JSONObject | str | bytes,
) -> Dict[str, str]:
    """Robustly extract YAML text from various agent output formats."""
    text = ""
    try:
        if isinstance(writer_output, dict):
            text = writer_output.get("yaml_text") or writer_output.get("generated_yaml") or ""
        elif isinstance(writer_output, bytes):
            text = writer_output.decode()
        elif hasattr(writer_output, "yaml_text"):
            text = getattr(writer_output, "yaml_text") or ""
        elif hasattr(writer_output, "generated_yaml"):
            text = getattr(writer_output, "generated_yaml") or ""
        elif hasattr(writer_output, "model_dump"):
            dumped = writer_output.model_dump()
            if isinstance(dumped, dict):
                text = dumped.get("yaml_text") or dumped.get("generated_yaml") or ""
        else:
            text = str(writer_output)
    except Exception:
        text = str(writer_output)

    if "```" in text:
        import re

        match = re.search(r"```(?:yaml|yml)?\n(.*)\n```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    if not ("version:" in text or "steps:" in text):
        try:
            repr_text = str(writer_output)
            if "generated_yaml" in repr_text:
                start = repr_text.find('generated_yaml:"') + len('generated_yaml:"')
                end = repr_text.rfind('"')
                if start < end:
                    text = repr_text[start:end]
        except Exception:
            pass

    return {"yaml_text": text or "", "generated_yaml": text or ""}


async def capture_validation_report(
    report: PydanticBaseModel | JSONObject,
) -> JSONObject:
    """Capture ValidationReport-like payload into context-friendly dict."""
    try:
        if hasattr(report, "model_dump"):
            report_dict = report.model_dump()
        elif isinstance(report, dict):
            report_dict = report
        else:
            report_dict = {}
        return {
            "validation_report": report_dict,
            "yaml_is_valid": bool(report_dict.get("is_valid", False)),
        }
    except Exception:
        return {"validation_report": {}, "yaml_is_valid": False}


async def check_user_confirmation(
    _out: str | JSONObject | None = None,
    ctx: Optional[DomainBaseModel] = None,
    *,
    user_input: Optional[str] = None,
    **_kwargs: Any,
) -> str:
    """Map free-form user input to 'approved'/'denied' branch keys."""
    _ = ctx  # unused
    text_val: Any = user_input if user_input is not None else _out
    try:
        text = "" if text_val is None else str(text_val)
    except Exception:
        text = ""
    norm = text.strip().lower()
    if norm == "" or norm in {"y", "yes"}:
        return "approved"
    return "denied"


async def compute_validity_key(
    _x: JSONObject | None = None, *, context: Optional[DomainBaseModel] = None
) -> str:
    try:
        val = bool(getattr(context, "yaml_is_valid", False))
    except Exception:
        try:
            if isinstance(context, dict):
                val = bool(context.get("yaml_is_valid", False))
            else:
                val = False
        except Exception:
            val = False
    return "valid" if val else "invalid"


def always_valid_key(
    _out: JSONObject | str | None = None, ctx: Optional[DomainBaseModel] = None
) -> str:
    """Return 'valid' unconditionally (used after successful repair)."""
    return "valid"


# --- In-memory YAML validation skill ---


async def validation_report_to_flag(
    report: PydanticBaseModel | JSONObject,
) -> JSONObject:
    """Return a dict with yaml_is_valid based on a ValidationReport-like input."""
    try:
        if isinstance(report, dict):
            val = bool(report.get("is_valid", False) or report.get("yaml_is_valid", False))
        else:
            val = bool(getattr(report, "is_valid", False))
    except Exception:
        val = False
    # Return the flag in output so the immediate next conditional can read it
    return {"yaml_is_valid": val}


async def extract_validation_errors(
    report: PydanticBaseModel | JSONObject,
    *,
    context: Optional[DomainBaseModel] = None,
) -> JSONObject:
    """Extract error messages from a ValidationReport-like input for repair loops.

    Also returns the current yaml_is_valid flag (when available) so that the
    subsequent ValidityBranch can read a decisive signal from the immediate
    previous output.
    """
    errors: List[str] = []
    try:
        # Prioritize the full validation report from context, as the `report` arg might be a summarized flag.
        report_source = None
        if hasattr(context, "validation_report"):
            report_source = getattr(context, "validation_report")
        elif hasattr(context, "errors"):
            report_source = context
        else:
            report_source = report

        report_dict: JSONObject
        if isinstance(report_source, dict):
            report_dict = report_source
        elif (
            report_source is not None
            and hasattr(report_source, "model_dump")
            and callable(getattr(report_source, "model_dump"))
        ):
            report_dict = report_source.model_dump()
        else:
            report_dict = {}
        for finding in report_dict.get("errors", []) or []:
            msg = finding.get("message") if isinstance(finding, dict) else None
            if msg:
                errors.append(str(msg))
    except Exception:
        errors = []
    import json as _json

    # Respect the explicit validity flag from the context as the source of truth
    is_valid = False
    try:
        is_valid = bool(getattr(context, "yaml_is_valid", False))
    except Exception:
        is_valid = False

    result: JSONObject = {
        "validation_errors": _json.dumps(errors),
        "yaml_is_valid": is_valid,
    }
    return result


# --- HITL helper: interpret user confirmation into branch key ---


def select_validity_branch(
    _out: JSONObject | str | None = None,
    ctx: Optional[DomainBaseModel] = None,
    **kwargs: Any,
) -> str:
    """Return 'valid' or 'invalid' using safe shape guard first, then explicit flags.

    Order:
    1) If YAML shape shows unmatched inline list on previous output or context, return 'invalid'
    2) If previous output dict carries 'yaml_is_valid', respect it
    3) Else if context.yaml_is_valid is present, respect it
    4) Else default to 'valid'
    """
    context = kwargs.get("context", ctx)

    def _shape_invalid(text: Any) -> bool:
        if not isinstance(text, str) or "steps:" not in text:
            return False
        try:
            line = text.split("steps:", 1)[1].splitlines()[0]
        except Exception:
            line = ""
        return ("[" in line and "]" not in line) and ("[]" not in line)

    try:
        val = None
        if isinstance(_out, dict) and "yaml_is_valid" in _out:
            try:
                val = bool(_out.get("yaml_is_valid"))
            except Exception:
                val = None
        print(
            "[SVB] out_type=",
            type(_out).__name__,
            " out_keys=",
            (list(_out.keys()) if isinstance(_out, dict) else None),
            " out_valid=",
            val,
            " ctx_flag=",
            (getattr(context, "yaml_is_valid", None) if context is not None else None),
            " ctx_has=",
            (hasattr(context, "yaml_is_valid") if context is not None else False),
            " ctx_type=",
            (type(context).__name__ if context is not None else None),
            " ctx_validation_report=",
            (hasattr(context, "validation_report") if context is not None else False),
        )
    except Exception:
        pass

    # 1) Early shape guard from previous output and context
    # Note: We don't discard _out here anymore, as the output flag should take priority
    try:
        if isinstance(_out, dict):
            yt0 = _out.get("yaml_text") or _out.get("generated_yaml")
            if _shape_invalid(yt0):
                return "invalid"
        elif isinstance(_out, str) and _shape_invalid(_out):
            return "invalid"
    except Exception:
        pass
    try:
        yt_ctx = getattr(context, "yaml_text", None)
        if _shape_invalid(yt_ctx):
            return "invalid"
    except Exception:
        pass

    # 2) Previous output signal
    try:
        if isinstance(_out, dict) and "yaml_is_valid" in _out:
            return "valid" if bool(_out.get("yaml_is_valid")) else "invalid"
    except Exception:
        pass

    # 3) Context flag
    try:
        if hasattr(context, "yaml_is_valid"):
            return "valid" if bool(getattr(context, "yaml_is_valid")) else "invalid"
    except Exception:
        pass
    if isinstance(context, dict) and "yaml_is_valid" in context:
        return "valid" if bool(context.get("yaml_is_valid")) else "invalid"

    # 4) Default to valid
    return "valid"


# --- Compute branch key from context validity (top-level importable) ---


def select_by_yaml_shape(
    _out: JSONObject | str | None = None,
    ctx: Optional[DomainBaseModel] = None,
    **kwargs: Any,
) -> str:
    """Return 'invalid' only for unmatched inline list on steps:

    Accepts both positional (output, context) and kw-only 'context'.
    """
    context = kwargs.get("context", ctx)
    """Heuristic selector to catch a very specific malformed YAML shape.

    - Returns 'invalid' only when the line after 'steps:' contains an opening '['
      without a matching closing ']' (e.g., "steps: ["), which is a common
      transient error pattern in early drafts.
    - Treats "steps: []" and other balanced inline lists as valid; also treats
    - normal block lists as valid.
    - Falls back to checking context.yaml_is_valid when available.
    """

    def _eval(text: str) -> str:
        parts = text.split("steps:", 1)
        if len(parts) == 2:
            line = parts[1].splitlines()[0]
            if "[" in line and "]" not in line and "[]" not in line:
                return "invalid"
        return "valid"

    try:
        prev = None
        if isinstance(_out, dict):
            prev = _out.get("yaml_text") or _out.get("generated_yaml")
        elif isinstance(_out, str):
            prev = _out
        ctx_text = None
        try:
            ctx_text = getattr(context, "yaml_text", None)
        except Exception:
            ctx_text = None
        print(
            "[SBYS] prev_is_dict=",
            isinstance(_out, dict),
            "ctx_flag=",
            (getattr(context, "yaml_is_valid", None) if context is not None else None),
            "prev_head=",
            (str(prev)[:30] if isinstance(prev, str) else None),
            "ctx_head=",
            (str(ctx_text)[:30] if isinstance(ctx_text, str) else None),
        )
    except Exception:
        pass

    try:
        if isinstance(_out, dict):
            val = _out.get("yaml_text") or _out.get("generated_yaml")
            if isinstance(val, str):
                res = _eval(val)
                if res == "invalid":
                    return res
        elif isinstance(_out, str):
            res = _eval(_out)
            if res == "invalid":
                return res
    except Exception:
        pass

    # Evaluate context.yaml_text shape next; this is authoritative for YAML shape
    try:
        yt_ctx = getattr(context, "yaml_text", None)
        if isinstance(yt_ctx, str):
            res = _eval(yt_ctx)
            if res == "invalid":
                return res
            # If shape looks valid, prefer 'valid' without relying on flags
            return "valid"
    except Exception:
        pass

    # Respect explicit context validity when provided
    try:
        if hasattr(context, "yaml_is_valid"):
            return "valid" if bool(getattr(context, "yaml_is_valid")) else "invalid"
    except Exception:
        pass
    if isinstance(context, dict) and "yaml_is_valid" in context:
        return "valid" if bool(context.get("yaml_is_valid")) else "invalid"

    # Fallback to context.yaml_text heuristic
    try:
        yt = getattr(context, "yaml_text", None)
        if isinstance(yt, str):
            res = _eval(yt)
            if res == "invalid":
                return res
    except Exception:
        pass
    return "valid"


async def shape_to_validity_flag(*, context: Optional[DomainBaseModel] = None) -> JSONObject:
    """Return {'yaml_is_valid': bool} based on a quick YAML shape heuristic.

    - False only when the 'steps:' line contains an opening '[' without a closing ']'.
    - True otherwise. This does not replace the validator; it just seeds a sensible default
      for the immediate conditional branch when validator behavior is mocked.
    """
    try:
        yt = getattr(context, "yaml_text", None)
    except Exception:
        yt = None
    if isinstance(yt, str) and "steps:" in yt:
        try:
            line = yt.split("steps:", 1)[1].splitlines()[0]
        except Exception:
            line = ""
        if "[" in line and "]" not in line:
            _out: JSONObject = {"yaml_is_valid": False, "yaml_text": yt}
            try:
                gy = getattr(context, "generated_yaml", None)
                if isinstance(gy, str):
                    _out["generated_yaml"] = gy
            except Exception:
                pass
            return _out
    # If we have any YAML-like structure with balanced inline list or block lists, treat as valid
    if isinstance(yt, str):
        try:
            after = yt.split("steps:", 1)[1]
            first = after.splitlines()[0]
            if (
                "[]" in first
                or ("[" in first and "]" in first)
                or not ("[" in first and "]" not in first)
            ):
                _out2: JSONObject = {"yaml_is_valid": True, "yaml_text": yt}
                try:
                    gy = getattr(context, "generated_yaml", None)
                    if isinstance(gy, str):
                        _out2["generated_yaml"] = gy
                except Exception:
                    pass
                return _out2
        except Exception:
            pass
    _out3: JSONObject = {"yaml_is_valid": True}
    if isinstance(yt, str):
        _out3["yaml_text"] = yt
    try:
        gy = getattr(context, "generated_yaml", None)
        if isinstance(gy, str):
            _out3["generated_yaml"] = gy
    except Exception:
        pass
    return _out3
