# Flujo Middleware API Guide

The **Middleware API** (`flujo.client.TaskClient`) allows external applications to inspect, manage, and interact with Flujo workflows.

It connects directly to your **State Backend** (Postgres or SQLite), enabling you to build:
*   **Custom UIs** (Dashboards, Chat interfaces, Approval portals).
*   **REST APIs** (FastAPI, Flask wrappers).
*   **Resilient Workers** (Cron jobs that remember where they left off).

## 1. Initialization

The `TaskClient` is your gateway. It requires a storage backend to read/write state.

```python
from flujo.client import TaskClient
from flujo.state.backends.sqlite import SQLiteBackend
# from flujo.state.backends.postgres import PostgresBackend

# 1. Connect to your state
# backend = PostgresBackend("postgresql://...") 
backend = SQLiteBackend("flujo_ops.db")

# 2. Initialize the client
client = TaskClient(backend=backend)
```

---

## 2. Core Capabilities

### A. Listing Tasks (The "Inbox" Pattern)
Retrieve a list of workflows based on status or metadata. This is perfect for building "Task Inboxes" or "Admin Dashboards."

**Method:** `client.list_tasks()`

```python
# Get all paused tasks waiting for human input
pending_tasks = await client.list_tasks(
    status="paused",
    limit=50
)

# Advanced: Filter by custom metadata tags
# (Useful for routing tasks to specific teams)
legal_reviews = await client.list_tasks(
    status="paused",
    metadata_filter={"department": "legal", "priority": "high"}
)

for task in legal_reviews:
    print(f"{task.run_id} - Started: {task.created_at}")
```

### B. Inspecting Details (The "Context" Pattern)
Before a human makes a decision, they need context. Fetch the full state of a specific workflow to see what the AI has done so far.

**Method:** `client.get_task(run_id)`

```python
task = await client.get_task("run_12345")

# 1. What did the AI ask? (The pause message)
print(f"AI Request: {task.last_prompt}")

# 2. Access data generated in previous steps
email_draft = task.context_snapshot.get("email_draft")
print(f"Draft Content: {email_draft}")

# 3. View execution history (costs, retries, errors)
print(f"Steps taken: {len(task.step_history)}")
if task.status == "failed":
    print(f"Error: {task.error_message}")
```

### C. Resuming Workflows (The "Action" Pattern)
When a human or external system provides input, use this to wake up the workflow.

**Method:** `client.resume_task(run_id, pipeline, input_data)`

```python
from my_project.pipeline import pipeline_def # Your Pipeline object

# Resume the workflow with new data
result = await client.resume_task(
    run_id="run_12345",
    pipeline=pipeline_def,
    input_data="Approved with edits: Change the greeting to 'Hi Team'."
)

print("Workflow finished!")
print(f"Final Output: {result.output}")
```

### D. System State (The "Watermark" Pattern)
Store global state that exists *outside* of specific runs. This is useful for connectors (e.g., "When did we last scrape this API?").

**Methods:** `get_system_state`, `set_system_state`

```python
# Check the last time we ran the nightly sync
last_sync = await client.get_system_state("nightly_sync_timestamp")

if not last_sync:
    # First run logic...
    pass

# Update the state
await client.set_system_state(
    key="nightly_sync_timestamp", 
    value={"timestamp": "2023-10-27T10:00:00Z", "items_processed": 500}
)
```

---

## 3. Common Recipes

### Recipe 1: The Approval Queue API
A simple FastAPI endpoint that lets a frontend fetch tasks awaiting approval.

```python
@app.get("/approvals")
async def get_approvals(department: str):
    """Get tasks paused and assigned to a specific department."""
    tasks = await client.list_tasks(
        status="paused",
        metadata_filter={"assigned_to": department}
    )
    
    # Return a simplified view for the frontend
    return [
        {
            "id": t.run_id,
            "title": t.metadata.get("title", "Untitled Task"),
            "waiting_since": t.updated_at
        }
        for t in tasks
    ]
```

### Recipe 2: The "Chat with Data" Interface
Using Flujo to power a chat interface where the AI can pause to ask clarifying questions.

```python
@app.post("/chat/{conversation_id}")
async def chat_turn(conversation_id: str, user_message: str):
    try:
        # Try to resume an existing conversation
        result = await client.resume_task(
            run_id=conversation_id,
            pipeline=chat_pipeline,
            input_data=user_message
        )
    except TaskNotFoundError:
        # Start a new conversation if ID doesn't exist
        # (Note: TaskClient manages existing tasks; Flujo runner starts new ones)
        runner = Flujo(chat_pipeline, state_backend=backend)
        result = await runner.run_async(user_message, run_id=conversation_id)

    # Check if AI finished or paused to ask a question back
    if result.status == "completed":
        return {"type": "answer", "text": result.output}
    elif result.status == "paused":
        # The AI hit a HumanInTheLoopStep
        ctx = result.final_pipeline_context
        question = ctx.pause_message
        return {"type": "question", "text": question}
```

### Recipe 3: Robust Cron Job (ETL)
A script that runs every hour but doesn't re-process data if it crashes halfway through.

```python
async def run_etl():
    # Check watermark
    state = await client.get_system_state("etl_cursor")
    last_id = state.value.get("last_id", 0) if state else 0
    
    # Define pipeline input
    payload = {"start_id": last_id, "limit": 100}
    
    # Check if a previous run failed and needs retry
    failed_runs = await client.list_tasks(
        status="failed", 
        metadata_filter={"job_type": "etl"}
    )
    
    if failed_runs:
        # Resume the failed run (retry logic handled by pipeline)
        print(f"Retrying failed run {failed_runs[0].run_id}...")
        await client.resume_task(failed_runs[0].run_id, etl_pipeline, payload)
    else:
        # Start new run
        runner = Flujo(etl_pipeline, state_backend=backend)
        await runner.run_async(payload, run_id=f"etl_{int(time.time())}")
        
        # Update watermark
        await client.set_system_state("etl_cursor", {"last_id": last_id + 100})

```

---

## 4. Best Practices

1.  **Inject Dependencies:** Do not hardcode database URIs in your API code. Load them from `os.environ` or `flujo.toml`.
2.  **Tag Your Runs:** When starting a pipeline, add metadata tags. This makes the `TaskClient` much more powerful later.
    ```python
    # When starting:
    task = Task(prompt="...", metadata={"user_id": "u_123", "type": "report"})
    await runner.run_async(task)
    
    # Later via Client:
    await client.list_tasks(metadata_filter={"user_id": "u_123"})
    ```
3.  **Handle Async:** `TaskClient` methods are `async`. Ensure your web framework (FastAPI/Quart) or script supports `await`.
4.  **Background Execution:** `resume_task` can take time (it runs the agent). In a web request, always wrap this in a background job (FastAPI `BackgroundTasks`, Celery, or Cloud Run Jobs) to avoid timeouts.
