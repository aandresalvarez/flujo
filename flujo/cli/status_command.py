from __future__ import annotations

from typing import Any, Optional

import click
import typer
from typing_extensions import Annotated

from flujo.type_definitions.common import JSONObject
from .exit_codes import EX_CONFIG_ERROR, EX_OK, EX_RUNTIME_ERROR


def status(
    format: Annotated[
        str,
        typer.Option(
            help="Output format",
            show_default=True,
            click_type=click.Choice(["text", "json"], case_sensitive=False),
        ),
    ] = "text",
    no_network: Annotated[
        bool,
        typer.Option(
            "--no-network",
            help="Skip live network checks (presence-only)",
        ),
    ] = False,
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            help="Per-check timeout in seconds for pings",
            show_default=True,
        ),
    ] = 2.0,
) -> None:
    """Status summary focusing on AI providers and SQLite configuration.

    - Providers: enabled if corresponding API key present.
    - SQLite: configured if state URI points at a sqlite path; skipped for memory.
    """
    import json
    import os as _os
    import traceback as _tb
    import time as _time

    # Prepare base payload
    payload: JSONObject = {
        "command": "status",
        "timestamp_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "providers": [],
    }

    # Provider presence checks; when network allowed, attempt lightweight model listing
    try:
        from flujo.infra.settings import get_settings as _get_settings

        s = _get_settings()
        providers: list[JSONObject] = []

        def _ping_openai(key: Optional[Any]) -> JSONObject:
            info: JSONObject = {"name": "openai", "enabled": bool(key)}
            if not key:
                info["status"] = "MISSING"
                return info
            if no_network:
                info["status"] = "OK"
                return info
            try:
                import urllib.request as _rq

                start = _time.perf_counter()
                req = _rq.Request(
                    "https://api.openai.com/v1/models",
                    headers={
                        "Authorization": f"Bearer {key.get_secret_value() if hasattr(key, 'get_secret_value') else str(key)}",
                        "Content-Type": "application/json",
                    },
                )
                with _rq.urlopen(req, timeout=float(timeout)) as resp:  # nosec - endpoint fixed
                    latency = int((_time.perf_counter() - start) * 1000)
                    code = getattr(resp, "status", 200)
                    if code == 200:
                        info.update({"status": "OK", "latency_ms": latency})
                    else:
                        info.update({"status": f"HTTP_{code}", "latency_ms": latency})
            except Exception as e:  # noqa: BLE001
                # Classify common HTTP errors
                status = "UNREACHABLE"
                if hasattr(e, "code"):
                    c = int(getattr(e, "code"))
                    status = {
                        401: "INVALID_KEY",
                        403: "FORBIDDEN",
                        429: "RATE_LIMITED",
                    }.get(c, f"HTTP_{c}")
                info.update({"status": status, "message": str(e)})
            return info

        def _ping_anthropic(key: Optional[Any]) -> JSONObject:
            info: JSONObject = {"name": "anthropic", "enabled": bool(key)}
            if not key:
                info["status"] = "MISSING"
                return info
            if no_network:
                info["status"] = "OK"
                return info
            try:
                import urllib.request as _rq

                start = _time.perf_counter()
                req = _rq.Request(
                    "https://api.anthropic.com/v1/models",
                    headers={
                        "x-api-key": key.get_secret_value()
                        if hasattr(key, "get_secret_value")
                        else str(key),
                        "anthropic-version": "2023-06-01",
                    },
                )
                with _rq.urlopen(req, timeout=float(timeout)) as resp:  # nosec - known URL
                    latency = int((_time.perf_counter() - start) * 1000)
                    code = getattr(resp, "status", 200)
                    if code == 200:
                        info.update({"status": "OK", "latency_ms": latency})
                    else:
                        info.update({"status": f"HTTP_{code}", "latency_ms": latency})
            except Exception as e:  # noqa: BLE001
                status = "UNREACHABLE"
                c = getattr(e, "code", None)
                if c is not None:
                    status = {401: "INVALID_KEY", 403: "FORBIDDEN", 429: "RATE_LIMITED"}.get(
                        int(c), f"HTTP_{int(c)}"
                    )
                info.update({"status": status, "message": str(e)})
            return info

        def _ping_gemini(key: Optional[Any]) -> JSONObject:
            info: JSONObject = {"name": "gemini", "enabled": bool(key)}
            if not key:
                info["status"] = "MISSING"
                return info
            if no_network:
                info["status"] = "OK"
                return info
            try:
                import urllib.request as _rq
                import urllib.parse as _up

                start = _time.perf_counter()
                query = _up.urlencode(
                    {
                        "key": key.get_secret_value()
                        if hasattr(key, "get_secret_value")
                        else str(key)
                    }
                )
                url = f"https://generativelanguage.googleapis.com/v1/models?{query}"
                with _rq.urlopen(url, timeout=float(timeout)) as resp:  # nosec - known URL
                    latency = int((_time.perf_counter() - start) * 1000)
                    code = getattr(resp, "status", 200)
                    if code == 200:
                        info.update({"status": "OK", "latency_ms": latency})
                    else:
                        info.update({"status": f"HTTP_{code}", "latency_ms": latency})
            except Exception as e:  # noqa: BLE001
                status = "UNREACHABLE"
                c = getattr(e, "code", None)
                if c is not None:
                    status = {401: "INVALID_KEY", 403: "FORBIDDEN", 429: "RATE_LIMITED"}.get(
                        int(c), f"HTTP_{int(c)}"
                    )
                info.update({"status": status, "message": str(e)})
            return info

        providers.append(_ping_openai(getattr(s, "openai_api_key", None)))
        providers.append(_ping_anthropic(getattr(s, "anthropic_api_key", None)))
        providers.append(_ping_gemini(getattr(s, "google_api_key", None)))
        payload["providers"] = providers
    except Exception as e:
        # Configuration errors should map to config exit code
        typer.secho(
            f"Failed to read provider settings: {type(e).__name__}: {e}", fg=typer.colors.RED
        )
        if _os.environ.get("FLUJO_CLI_VERBOSE") == "1":
            typer.echo(_tb.format_exc(), err=True)
        raise typer.Exit(EX_CONFIG_ERROR)

    # State backend configuration insight (SQLite and PostgreSQL)
    try:
        from ..infra.config_manager import get_state_uri as _get_state_uri
        from ..infra.config_manager import get_config_manager as _get_cfg_mgr
        from ..state.sqlite_uri import normalize_sqlite_path as _norm_sqlite
        from urllib.parse import urlparse as _urlparse
        from pathlib import Path
        import sqlite3 as _sql

        uri = _get_state_uri(force_reload=True)
        try:
            cfg_mgr = _get_cfg_mgr(force_reload=False)
            config_path = getattr(cfg_mgr, "config_path", None)
        except Exception:
            config_path = None
        sqlite_info: JSONObject = {"configured": False}
        postgres_info: JSONObject = {"configured": False}
        if uri:
            uri_lower = uri.strip().lower()
            # Memory-like forms => not configured for SQLite or Postgres
            if uri_lower in {"memory", "memory://", "mem://", "inmemory://"}:
                sqlite_info = {"configured": False}
                postgres_info = {"configured": False}
            else:
                parsed = _urlparse(uri)
                scheme_lower = parsed.scheme.lower()
                if scheme_lower.startswith("sqlite"):
                    try:
                        db_path = _norm_sqlite(
                            uri, Path.cwd(), config_dir=config_path.parent if config_path else None
                        )
                        sqlite_info = {
                            "configured": True,
                            "path": db_path.as_posix(),
                        }
                    except Exception as pe:
                        sqlite_info = {
                            "configured": False,
                            "error": f"Failed to parse SQLite path: {type(pe).__name__}: {pe}",
                        }
                elif scheme_lower in {"postgres", "postgresql"}:
                    # Extract connection info (without password for security)
                    host = parsed.hostname or "localhost"
                    port = parsed.port or 5432
                    database = parsed.path.lstrip("/") if parsed.path else None
                    user = parsed.username
                    postgres_info = {
                        "configured": True,
                        "host": host,
                        "port": port,
                        "database": database,
                        "user": user,
                        "uri_scheme": scheme_lower,
                    }
                else:
                    # Unknown scheme => not our concern in MVP
                    sqlite_info = {"configured": False}
                    postgres_info = {"configured": False}
        payload["sqlite"] = sqlite_info
        payload["postgres"] = postgres_info

        # Last runs summary when SQLite is configured and file exists
        try:
            history: JSONObject = {"available": False, "items": []}
            if sqlite_info.get("configured") and sqlite_info.get("path"):
                dbp = Path(str(sqlite_info["path"]))
                if dbp.exists():
                    try:
                        ro_uri = f"file:{dbp.as_posix()}?mode=ro"
                        conn = _sql.connect(ro_uri, uri=True)
                        try:
                            cur = conn.cursor()
                            cur.execute(
                                "SELECT run_id, pipeline_name, status, created_at, updated_at, execution_time_ms, error_message FROM runs ORDER BY created_at DESC LIMIT 3"
                            )
                            items = []
                            for row in cur.fetchall() or []:
                                items.append(
                                    {
                                        "id": row[0],
                                        "pipeline": row[1],
                                        "status": row[2],
                                        "started_at": row[3],
                                        "ended_at": row[4],
                                        "duration_ms": row[5],
                                        "error": row[6],
                                    }
                                )
                            history.update({"available": True, "items": items, "total": len(items)})
                        finally:
                            conn.close()
                    except Exception:
                        # If schema absent/corrupt, just mark unavailable
                        history = {"available": False, "items": []}
            payload["history"] = history
        except Exception:
            # Never fail status due to history lookup
            payload["history"] = {"available": False, "items": []}
    except Exception as e:
        typer.secho(f"Failed to inspect state URI: {type(e).__name__}: {e}", fg=typer.colors.RED)
        if _os.environ.get("FLUJO_CLI_VERBOSE") == "1":
            typer.echo(_tb.format_exc(), err=True)
        raise typer.Exit(EX_RUNTIME_ERROR)

    # Emit output
    if (format or "text").lower() == "json":
        try:
            from pydantic import BaseModel as _BM

            if isinstance(payload, _BM):
                try:
                    payload_data = payload.model_dump(mode="json")
                except TypeError:
                    payload_data = payload.model_dump()
            else:
                payload_data = payload
        except Exception:
            payload_data = payload
        typer.echo(json.dumps(payload_data))
    else:
        try:
            from rich.console import Console as _Console

            console = _Console()
        except ModuleNotFoundError:
            console = None
        if console is None:

            class _PlainConsole:
                def print(self, msg: object, *args: object, **kwargs: object) -> None:
                    from .helpers import print_rich_or_typer as _prt

                    _prt(str(msg))

            console = _PlainConsole()  # type: ignore[assignment]
        # Providers line
        pv_summ = ", ".join(
            f"{p['name']}: {'ENABLED' if p.get('enabled') else 'disabled'}"
            for p in payload["providers"]
        )
        if console is not None:
            console.print(f"Providers: {pv_summ}")
        else:
            from .helpers import print_rich_or_typer

            print_rich_or_typer(f"Providers: {pv_summ}")
        # SQLite line
        sqlite_info = payload.get("sqlite", {})
        if sqlite_info.get("configured"):
            if console is not None:
                console.print(f"SQLite: configured ({sqlite_info.get('path')})")
            else:
                from .helpers import print_rich_or_typer

                print_rich_or_typer(f"SQLite: configured ({sqlite_info.get('path')})")
        else:
            if console is not None:
                console.print("SQLite: not configured (memory or absent)")
            else:
                from .helpers import print_rich_or_typer

                print_rich_or_typer("SQLite: not configured (memory or absent)")

        # History preview
        hist = payload.get("history", {})
        if hist.get("available") and hist.get("items"):
            items = hist.get("items", [])
            preview = "; ".join(
                f"{it.get('started_at', '?')} {it.get('status', '?')}" for it in items
            )
            if console is not None:
                console.print(f"History: {len(items)} runs: {preview}")
            else:
                from .helpers import print_rich_or_typer

                print_rich_or_typer(f"History: {len(items)} runs: {preview}")

    raise typer.Exit(EX_OK)


__all__ = ["status"]
