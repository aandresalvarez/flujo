import subprocess
import os
import uuid
import sys


def run_cli(cmd, env=None):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    return result.stdout


def main():
    db_path = f"ops_test_{uuid.uuid4().hex}.db"
    schema_sql = """
    CREATE TABLE workflow_state (
        run_id TEXT PRIMARY KEY,
        pipeline_id TEXT,
        pipeline_name TEXT,
        pipeline_version TEXT,
        current_step_index INTEGER,
        pipeline_context TEXT,
        last_step_output TEXT,
        step_history TEXT,
        status TEXT,
        created_at TEXT,
        updated_at TEXT,
        total_steps INTEGER,
        error_message TEXT,
        execution_time_ms INTEGER,
        memory_usage_mb REAL
    );
    """
    insert_sqls = [
        f"""INSERT INTO workflow_state VALUES ('run_{i}', 'test-pid-run_{i}', 'pipeline_{i}', '1.0', 1, '{{}}', NULL, '[]', '{"completed" if i % 2 == 0 else "failed"}', '2025-07-23T20:50:00', '2025-07-23T20:50:00', 2, NULL, 100, 10.0);"""
        for i in range(5)
    ]
    subprocess.run(["sqlite3", db_path, schema_sql], check=True)
    for insert_sql in insert_sqls:
        subprocess.run(["sqlite3", db_path, insert_sql], check=True)

    env = os.environ.copy()
    env["FLUJO_STATE_URI"] = f"sqlite:///{os.path.abspath(db_path)}"
    output = run_cli("flujo lens list", env=env)
    for i in range(5):
        assert f"run_{i}" in output, f"run_{i} not found in CLI output"
    print("[SUCCESS] All runs found in CLI output.")


if __name__ == "__main__":
    main()
