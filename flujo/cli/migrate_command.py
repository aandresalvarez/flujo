from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import typer

from flujo.infra.config_manager import get_state_uri
from flujo.utils.async_bridge import run_sync


def migrate(dry_run: bool = False, target_version: Optional[int] = None) -> None:
    """Apply PostgreSQL migrations for the configured state backend."""

    uri = get_state_uri(force_reload=True)
    parsed = urlparse(uri or "")
    scheme = (parsed.scheme or "").lower()
    if uri is None or scheme not in {"postgres", "postgresql"}:
        typer.secho("PostgreSQL state_uri required for migrations", fg=typer.colors.RED)
        raise typer.Exit(1)

    if target_version is not None and target_version < 0:
        typer.secho("target-version must be non-negative", fg=typer.colors.RED)
        raise typer.Exit(1)

    spec = importlib.util.find_spec("asyncpg")
    if spec is None:
        typer.secho(
            "asyncpg is required for migrations. Install with `pip install flujo[postgres]`.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    run_sync(_run_migrations(uri, dry_run=dry_run, target_version=target_version))


async def _run_migrations(uri: str, dry_run: bool, target_version: Optional[int]) -> None:
    import asyncpg

    migrations_dir = Path(__file__).resolve().parent.parent / "state" / "migrations"
    migration_files: List[Tuple[int, Path]] = []
    for path in migrations_dir.glob("*.sql"):
        stem = path.stem.split("_", 1)[0]
        if stem.isdigit():
            version = int(stem)
            if target_version is not None and version > target_version:
                continue
            migration_files.append((version, path))
    migration_files.sort(key=lambda item: item[0])

    conn = await asyncpg.connect(uri)
    try:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS flujo_schema_versions (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        applied_rows = await conn.fetch("SELECT version FROM flujo_schema_versions")
        applied_versions = {int(row["version"]) for row in applied_rows}
        pending = [(v, p) for v, p in migration_files if v not in applied_versions]
        if not pending:
            typer.echo("Database is already up-to-date.")
            return

        for version, path in pending:
            sql = path.read_text()
            if dry_run:
                typer.echo(f"[dry-run] Would apply migration {path.name}")
                continue
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    """
                    INSERT INTO flujo_schema_versions (version, applied_at)
                    VALUES ($1, NOW())
                    ON CONFLICT (version) DO NOTHING
                    """,
                    version,
                )
            typer.echo(f"Applied migration {path.name}")
    finally:
        await conn.close()
