from __future__ import annotations

from typing import Any

from flujo.type_definitions.common import JSONObject

from flujo.architect.states.common import skill_resolver, telemetry
from flujo.domain.base_model import BaseModel as _BaseModel
from flujo.domain.dsl import Pipeline, Step
from flujo.exceptions import InfiniteRedirectError, PausedException, PipelineAbortSignal


async def _select_yaml_text(_x: Any = None, *, context: _BaseModel | None = None) -> str:
    try:
        yt = getattr(context, "yaml_text", "") if context is not None else ""
        return yt if isinstance(yt, str) else ""
    except Exception:
        return ""


def build_validation_state() -> Pipeline[Any, Any]:
    """Validate YAML; repair and re-loop until valid."""
    reg = skill_resolver()
    try:
        validate_entry = reg.get("flujo.builtins.validate_yaml")
        if validate_entry and isinstance(validate_entry, dict):
            _validate = validate_entry["factory"]()
            validate: Step[Any, Any] = Step.from_callable(_validate, name="ValidateYAML")
        else:

            async def _validate_fallback(yt: Any) -> JSONObject:
                return {"is_valid": True}

            validate = Step.from_callable(_validate_fallback, name="ValidateYAML")
    except Exception:

        async def _validate_fallback(yt: Any) -> JSONObject:
            return {"is_valid": True}

        validate = Step.from_callable(_validate_fallback, name="ValidateYAML")

    try:
        capture_entry = reg.get("flujo.builtins.capture_validation_report")
        telemetry().info(f"[ArchitectSM] CaptureReport registry lookup: {capture_entry}")
        if capture_entry and isinstance(capture_entry, dict):
            _capture = capture_entry["factory"]()
            telemetry().info("[ArchitectSM] CaptureReport using registry function")
            capture: Step[Any, Any] = Step.from_callable(
                _capture, name="CaptureReport", updates_context=True
            )
        else:
            telemetry().info("[ArchitectSM] CaptureReport using fallback function")

            async def _capture_fallback(
                rep: Any, *, context: _BaseModel | None = None
            ) -> JSONObject:
                is_valid = True
                try:
                    telemetry().info(f"[ArchitectSM] CaptureReport FIRST fallback: rep={rep}")
                    if isinstance(rep, dict) and "is_valid" in rep:
                        is_valid = bool(rep.get("is_valid"))
                        msg = f"[ArchitectSM] CaptureReport: extracted is_valid={is_valid} from rep"
                        telemetry().info(msg)
                    if context is not None and hasattr(context, "yaml_is_valid"):
                        try:
                            setattr(context, "yaml_is_valid", is_valid)
                            msg = (
                                "[ArchitectSM] CaptureReport: "
                                f"set yaml_is_valid={is_valid} in context"
                            )
                            telemetry().info(msg)
                        except Exception as e:
                            telemetry().error(
                                f"[ArchitectSM] CaptureReport: Failed to set yaml_is_valid: {e}"
                            )
                except Exception:
                    pass
                return {"validation_report": rep, "yaml_is_valid": is_valid}

            capture = Step.from_callable(
                _capture_fallback, name="CaptureReport", updates_context=True
            )
    except Exception:

        async def _capture_fallback(rep: Any, *, context: _BaseModel | None = None) -> JSONObject:
            is_valid = True
            try:
                if isinstance(rep, dict):
                    if "is_valid" in rep:
                        is_valid = bool(rep.get("is_valid"))
                    elif "errors" in rep and rep.get("errors"):
                        is_valid = False
                telemetry().info(f"[ArchitectSM] CaptureReport: setting yaml_is_valid={is_valid}")
            except Exception as e:
                telemetry().error(f"[ArchitectSM] CaptureReport error: {e}")
                is_valid = True
            return {"validation_report": rep, "yaml_is_valid": is_valid}

        capture = Step.from_callable(_capture_fallback, name="CaptureReport", updates_context=True)

    async def _decide_next(_rep: Any = None, *, context: _BaseModel | None = None) -> JSONObject:
        valid = False
        try:
            if isinstance(_rep, dict) and "is_valid" in _rep:
                valid = bool(_rep.get("is_valid"))
        except Exception:
            pass
        if not valid and context is not None:
            try:
                valid = bool(getattr(context, "yaml_is_valid", False))
            except Exception:
                valid = False

        try:
            telemetry().info(f"[ArchitectSM] ValidationDecision: yaml_is_valid={valid}")
        except Exception:
            pass

        if valid:
            try:
                telemetry().info("[ArchitectSM] ValidationDecision -> DryRunOffer")
            except Exception:
                pass
            return {"next_state": "DryRunOffer"}

        try:
            telemetry().info("[ArchitectSM] ValidationDecision -> Validation (repair attempt)")
        except Exception:
            pass
        try:
            repair_entry = reg.get("flujo.builtins.repair_yaml_ruamel")
            if repair_entry and isinstance(repair_entry, dict):
                _repair = repair_entry["factory"]()
                repaired = await _repair(getattr(context, "yaml_text", ""))
                if isinstance(repaired, dict):
                    out: JSONObject = {**repaired}
                else:
                    out = {}
            else:
                out = {}
        except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
            raise
        except Exception:
            out = {}

        out["next_state"] = "Validation"
        return out

    decide_next = Step.from_callable(_decide_next, name="ValidationDecision", updates_context=True)

    return Pipeline.from_step(Step.from_callable(_select_yaml_text, name="SelectYAMLText")) >> (
        validate >> capture >> decide_next
    )
