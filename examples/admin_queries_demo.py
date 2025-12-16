#!/usr/bin/env python3
"""
Admin Queries Demo for Optimized SQLite State Backend

This script demonstrates the new admin query capabilities for monitoring
and managing workflow states in production environments.
"""

import asyncio
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flujo.state import SQLiteBackend


async def create_sample_workflows(backend: SQLiteBackend, count: int = 50) -> None:
    """Create sample workflows for demonstration purposes."""
    print(f"Creating {count} sample workflows...")

    pipeline_names = [
        "Data Processing Pipeline",
        "ML Training Pipeline",
        "ETL Pipeline",
        "Report Generation",
        "Data Validation",
    ]

    statuses = ["running", "completed", "failed", "paused", "cancelled"]
    error_messages = [
        "Connection timeout",
        "Invalid input data",
        "Memory limit exceeded",
        "API rate limit exceeded",
        "Database connection failed",
        None,  # Some workflows succeed
    ]

    for i in range(count):
        # Create realistic timestamps
        created_at = datetime.now(timezone.utc) - timedelta(
            hours=random.randint(0, 168),  # Up to 1 week ago
            minutes=random.randint(0, 60),
        )
        updated_at = created_at + timedelta(minutes=random.randint(1, 120))

        status = random.choice(statuses)
        pipeline_name = random.choice(pipeline_names)

        state = {
            "run_id": f"demo_run_{i:03d}",
            "pipeline_id": f"pipeline_{pipeline_name.lower().replace(' ', '_')}",
            "pipeline_name": pipeline_name,
            "pipeline_version": f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
            "current_step_index": random.randint(0, 10),
            "pipeline_context": {
                "input_file": f"data_{i}.csv",
                "batch_size": random.randint(100, 10000),
                "user_id": f"user_{random.randint(1, 100)}",
            },
            "last_step_output": {
                "processed_rows": random.randint(0, 50000),
                "errors": random.randint(0, 10),
            }
            if status in ["completed", "running"]
            else None,
            "status": status,
            "created_at": created_at,
            "updated_at": updated_at,
            "total_steps": random.randint(3, 15),
            "error_message": random.choice(error_messages) if status == "failed" else None,
            "execution_time_ms": random.randint(1000, 300000)
            if status in ["completed", "failed"]
            else None,
            "memory_usage_mb": random.uniform(10.0, 500.0)
            if status in ["completed", "failed"]
            else None,
        }

        await backend.save_state(f"demo_run_{i:03d}", state)

    print("Sample workflows created successfully!")


async def demonstrate_listing_workflows(backend: SQLiteBackend) -> None:
    """Demonstrate workflow listing with various filters."""
    print("\n" + "=" * 60)
    print("WORKFLOW LISTING DEMONSTRATIONS")
    print("=" * 60)

    # List all workflows (limited to 5 for demo)
    print("\n1. Recent workflows (limit 5):")
    workflows = await backend.list_workflows(limit=5)
    for wf in workflows:
        print(f"  {wf['run_id']}: {wf['pipeline_name']} ({wf['status']}) - {wf['updated_at']}")

    # List running workflows
    print("\n2. Currently running workflows:")
    running = await backend.list_workflows(status="running")
    for wf in running:
        print(f"  {wf['run_id']}: {wf['pipeline_name']} - Step {wf['current_step_index']}")

    # List failed workflows
    print("\n3. Failed workflows:")
    failed = await backend.list_workflows(status="failed")
    for wf in failed:
        print(f"  {wf['run_id']}: {wf['pipeline_name']}")

    # List workflows by pipeline
    print("\n4. Data Processing Pipeline workflows:")
    data_pipeline = await backend.list_workflows(pipeline_id="pipeline_data_processing_pipeline")
    for wf in data_pipeline:
        print(f"  {wf['run_id']}: {wf['status']} - {wf['updated_at']}")


async def demonstrate_statistics(backend: SQLiteBackend) -> None:
    """Demonstrate workflow statistics."""
    print("\n" + "=" * 60)
    print("WORKFLOW STATISTICS")
    print("=" * 60)

    stats = await backend.get_workflow_stats()

    print("\n📊 Overall Statistics:")
    print(f"  Total workflows: {stats['total_workflows']}")
    print(f"  Recent workflows (24h): {stats['recent_workflows_24h']}")
    print(f"  Average execution time: {stats['average_execution_time_ms']:.1f}ms")

    print("\n📈 Status Breakdown:")
    for status, count in stats["status_counts"].items():
        percentage = (count / stats["total_workflows"]) * 100
        print(f"  {status.capitalize()}: {count} ({percentage:.1f}%)")

    # Calculate success rate
    completed = stats["status_counts"].get("completed", 0)
    failed = stats["status_counts"].get("failed", 0)
    total_finished = completed + failed
    if total_finished > 0:
        success_rate = (completed / total_finished) * 100
        print(f"\n✅ Success Rate: {success_rate:.1f}%")


async def demonstrate_failed_workflow_analysis(backend: SQLiteBackend) -> None:
    """Demonstrate failed workflow analysis."""
    print("\n" + "=" * 60)
    print("FAILED WORKFLOW ANALYSIS")
    print("=" * 60)

    # Get failed workflows from last 24 hours
    failed_workflows = await backend.get_failed_workflows(hours_back=24)

    if not failed_workflows:
        print("No failed workflows in the last 24 hours.")
        return

    print(f"\n🔍 Found {len(failed_workflows)} failed workflows in the last 24 hours:")

    # Group by error message
    error_groups = {}
    for wf in failed_workflows:
        error = wf["error_message"] or "Unknown error"
        if error not in error_groups:
            error_groups[error] = []
        error_groups[error].append(wf)

    print("\n📋 Error Analysis:")
    for error, workflows in error_groups.items():
        print(f"\n  Error: {error}")
        print(f"  Occurrences: {len(workflows)}")
        for wf in workflows[:3]:  # Show first 3 examples
            print(f"    - {wf['run_id']} ({wf['pipeline_name']}) at {wf['updated_at']}")
        if len(workflows) > 3:
            print(f"    ... and {len(workflows) - 3} more")


async def demonstrate_cleanup_operations(backend: SQLiteBackend) -> None:
    """Demonstrate cleanup operations."""
    print("\n" + "=" * 60)
    print("CLEANUP OPERATIONS")
    print("=" * 60)

    # Show current count
    stats_before = await backend.get_workflow_stats()
    print(f"\n📊 Before cleanup: {stats_before['total_workflows']} workflows")

    # Simulate cleanup (in real usage, you'd use a longer period)
    # For demo, we'll use a very short period to show the concept
    print("\n🧹 Cleaning up workflows older than 1 hour...")
    deleted_count = await backend.cleanup_old_workflows(days_old=1 / 24)  # 1 hour

    if deleted_count > 0:
        print(f"✅ Deleted {deleted_count} old workflows")

        stats_after = await backend.get_workflow_stats()
        print(f"📊 After cleanup: {stats_after['total_workflows']} workflows")
    else:
        print("ℹ️  No workflows old enough to delete")


async def demonstrate_direct_sql_queries(backend: SQLiteBackend) -> None:
    """Demonstrate direct SQL queries for advanced analysis."""
    print("\n" + "=" * 60)
    print("DIRECT SQL QUERIES")
    print("=" * 60)

    import aiosqlite

    async with aiosqlite.connect(backend.db_path) as db:
        # Performance analysis
        print("\n🚀 Performance Analysis:")
        cursor = await db.execute("""
            SELECT
                pipeline_name,
                AVG(execution_time_ms) as avg_time,
                COUNT(*) as total_runs,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful
            FROM workflow_state
            WHERE execution_time_ms IS NOT NULL
            GROUP BY pipeline_name
            ORDER BY avg_time DESC
        """)

        rows = await cursor.fetchall()
        for row in rows:
            pipeline, avg_time, total, successful = row
            success_rate = (successful / total) * 100 if total > 0 else 0
            print(f"  {pipeline}: {avg_time:.0f}ms avg, {success_rate:.1f}% success rate")

        # Memory usage analysis
        print("\n💾 Memory Usage Analysis:")
        cursor = await db.execute("""
            SELECT
                pipeline_name,
                AVG(memory_usage_mb) as avg_memory,
                MAX(memory_usage_mb) as max_memory
            FROM workflow_state
            WHERE memory_usage_mb IS NOT NULL
            GROUP BY pipeline_name
            ORDER BY avg_memory DESC
        """)

        rows = await cursor.fetchall()
        for row in rows:
            pipeline, avg_memory, max_memory = row
            print(f"  {pipeline}: {avg_memory:.1f}MB avg, {max_memory:.1f}MB max")


async def demonstrate_monitoring_setup(backend: SQLiteBackend) -> None:
    """Demonstrate a monitoring setup for production use."""
    print("\n" + "=" * 60)
    print("PRODUCTION MONITORING SETUP")
    print("=" * 60)

    # Health check
    print("\n🏥 Health Check:")
    stats = await backend.get_workflow_stats()

    total = stats["total_workflows"]
    failed = stats["status_counts"].get("failed", 0)
    running = stats["status_counts"].get("running", 0)

    # Calculate metrics
    failure_rate = (failed / total) * 100 if total > 0 else 0
    recent_activity = stats["recent_workflows_24h"]

    # Check thresholds
    alerts = []

    if failure_rate > 10:
        alerts.append(f"🚨 High failure rate: {failure_rate:.1f}%")

    if running > 20:
        alerts.append(f"⚠️  Many running workflows: {running}")

    if recent_activity < 5:
        alerts.append(f"📉 Low recent activity: {recent_activity} workflows in 24h")

    if alerts:
        print("  Alerts:")
        for alert in alerts:
            print(f"    {alert}")
    else:
        print("  ✅ All systems operational")

    # Performance summary
    avg_exec_time = stats["average_execution_time_ms"]
    if avg_exec_time:
        if avg_exec_time > 60000:  # 1 minute
            print(f"  ⚠️  Slow average execution: {avg_exec_time / 1000:.1f}s")
        else:
            print(f"  ✅ Good performance: {avg_exec_time / 1000:.1f}s average")


async def main():
    """Main demonstration function."""
    print("🚀 SQLite State Backend Admin Queries Demo")
    print("=" * 60)

    # Initialize backend
    db_path = Path("demo_workflow_state.db")
    backend = SQLiteBackend(db_path)

    try:
        # Create sample data
        await create_sample_workflows(backend, count=30)

        # Run demonstrations
        await demonstrate_listing_workflows(backend)
        await demonstrate_statistics(backend)
        await demonstrate_failed_workflow_analysis(backend)
        await demonstrate_cleanup_operations(backend)
        await demonstrate_direct_sql_queries(backend)
        await demonstrate_monitoring_setup(backend)

        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print(f"📁 Database file: {db_path.absolute()}")
        print("\n💡 You can now explore the database directly with SQLite tools:")
        print(f"   sqlite3 {db_path}")

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        raise
    finally:
        # Clean up demo database
        if db_path.exists():
            db_path.unlink()
            print(f"\n🧹 Cleaned up demo database: {db_path}")


if __name__ == "__main__":
    asyncio.run(main())
