"""Example: Sharing resources across steps with `AppResources`.

This script corresponds to `docs/cookbook/using_resources.md`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from flujo import Flujo, Step, AppResources


class MyWebAppResources(AppResources):
    """Defines the shared resources for the app."""

    db_conn: MagicMock


class UserLookupAgent:
    async def run(self, user_id: int, *, resources: MyWebAppResources) -> str:
        print(f"AGENT: Looking up user {user_id}...")
        return resources.db_conn.get_user_by_id(user_id)


def build_runner() -> tuple[Flujo[int, str], MyWebAppResources]:
    resources = MyWebAppResources(db_conn=MagicMock())
    resources.db_conn.get_user_by_id.return_value = "Alice"

    pipeline = Step("lookup_user", UserLookupAgent())
    runner = Flujo(pipeline, resources=resources)
    return runner, resources


def main() -> None:
    runner, resources = build_runner()
    result = runner.run(123)
    resources.db_conn.get_user_by_id.assert_called_once_with(123)
    print(f"\n✅ Agent successfully used the database to find: {result.step_history[0].output}")


if __name__ == "__main__":
    main()
