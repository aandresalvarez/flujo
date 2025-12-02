"""Shared fixtures for integration tests."""

from __future__ import annotations

import importlib.util
import os

import pytest

from flujo.state.backends.postgres import PostgresBackend

# Check if asyncpg is available
_asyncpg_available = importlib.util.find_spec("asyncpg") is not None

# Check if testcontainers is available
_testcontainers_available = importlib.util.find_spec("testcontainers") is not None


@pytest.fixture
async def postgres_backend() -> PostgresBackend:
    """Create a PostgresBackend instance for testing.

    Uses testcontainers if available, otherwise requires FLUJO_TEST_POSTGRES_URI env var.
    """
    if not _asyncpg_available:
        pytest.skip("asyncpg not installed")

    if _testcontainers_available:
        try:
            from testcontainers.postgres import PostgresContainer

            with PostgresContainer("postgres:15") as postgres:
                dsn = postgres.get_connection_url().replace("postgresql://", "postgres://")
                backend = PostgresBackend(dsn, auto_migrate=True)
                yield backend
                await backend.shutdown()
        except Exception as e:
            pytest.skip(f"Failed to start testcontainers Postgres: {e}")
    else:
        test_uri = os.environ.get("FLUJO_TEST_POSTGRES_URI")
        if not test_uri:
            pytest.skip(
                "testcontainers not available and FLUJO_TEST_POSTGRES_URI not set. "
                "Install testcontainers or set FLUJO_TEST_POSTGRES_URI environment variable."
            )
        backend = PostgresBackend(test_uri, auto_migrate=True)
        yield backend
        await backend.shutdown()
