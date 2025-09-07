from __future__ import annotations

from typing import Any


def log_audit(event: str, **fields: Any) -> None:
    """Emit a lightweight audit log entry via telemetry.

    Format: "[audit] event=<event> key=value ..." using simple stringification.
    Intended for human/grep-friendly logs without imposing a strict schema.
    """
    try:
        from . import telemetry  # Local import to avoid early import side effects

        parts = [f"event={event}"]
        for k in sorted(fields.keys()):
            v = fields[k]
            try:
                parts.append(f"{k}={v}")
            except Exception:
                parts.append(f"{k}=<unrepr>")
        telemetry.logfire.info("[audit] " + " ".join(parts))
    except Exception:
        # Never raise from audit logging
        pass
