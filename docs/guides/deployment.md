# 🚀 Flujo Production Deployment Guide

This guide outlines the best practices for moving Flujo pipelines from a local development environment to a scalable, production-ready cloud architecture.

## The Golden Rule of Flujo Deployment

**Flujo is designed to be stateless in memory, but stateful in storage.**

1.  **The Container is Ephemeral:** Never rely on the container's memory or local filesystem to store run history. Containers scale up, down, and crash.
2.  **The Database is Truth:** All execution state (step history, variables, retries) must be persisted to a robust remote database (Postgres).

---

## 1. Database Configuration (Critical)

In development, you likely used SQLite (`.flujo/state.db`). In production, you **must** switch to PostgreSQL to handle concurrency and persistence across container restarts.

**`flujo.toml` configuration:**

```toml
# Production configuration
state_uri = "postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:5432/${DB_NAME}"

[settings]
# Disable debug features for performance
test_mode = false
```

*   **Recommended Providers:** Google Cloud SQL, AWS RDS, Neon, or Supabase.
*   **Migrations:** Ensure you run `flujo migrate` as part of your deployment pipeline (CD) to keep the database schema updated.

---

## 2. Choosing Your Architecture

Select the architecture that matches your agent's time horizon.

### Scenario A: The "Chatbot" (Synchronous, <60 seconds)
**Best for:** RAG, customer support bots, quick data extraction.
**Infrastructure:** Google Cloud Run (Service), AWS App Runner, or Kubernetes Deployment.

Wrap Flujo in a lightweight ASGI server (FastAPI/Uvicorn). Because Flujo removes heavy dependencies like `psutil`, it starts up instantly, making it ideal for "Scale to Zero" platforms.

```python
# main.py
from fastapi import FastAPI
from flujo import Flujo, Pipeline

app = FastAPI()
# Load pipeline once at startup
pipeline = Pipeline.from_yaml("pipeline.yaml")

@app.post("/chat")
async def chat(message: str):
    # Run synchronously (the user waits for the answer)
    runner = Flujo(pipeline)
    result = await runner.run_async(message)
    return {"response": result.output}
```

### Scenario B: The "Researcher" (Asynchronous, 1 min to 24 hours)
**Best for:** Deep research, report generation, batch processing.
**Infrastructure:** Google Cloud Run **Jobs**, AWS Fargate, or Modal.

Do not keep an HTTP connection open. Instead, trigger a job. This provides crash resilience; if the container dies, the platform restarts it, and Flujo resumes from the last saved step in Postgres.

```python
# worker.py
import os
import asyncio
from flujo import Flujo, Pipeline

async def main():
    # Get input from environment variables (passed by the Job orchestrator)
    input_data = os.getenv("AGENT_INPUT")
    run_id = os.getenv("AGENT_RUN_ID")
    
    pipeline = Pipeline.from_yaml("pipeline.yaml")
    
    # If run_id exists in Postgres, Flujo automatically RESUMES.
    # If not, it starts fresh.
    runner = Flujo(pipeline)
    await runner.run_async(input_data, run_id=run_id)

if __name__ == "__main__":
    asyncio.run(main())
```

### Scenario C: The "Human-in-the-Loop" (Multi-Day)
**Best for:** Approval workflows, email correspondence.
**Infrastructure:** Any of the above + Persistence.

You do not need a server running for 3 days waiting for a human.
1.  **Run:** Agent executes until it hits a `HumanInTheLoopStep`.
2.  **Pause:** Flujo saves state to Postgres and the process exits (saving money).
3.  **Resume:** When the user clicks "Approve" on your UI, trigger a new container with the `run_id` and the human input.

```python
# Resume logic via API
@app.post("/resume/{run_id}")
async def resume(run_id: str, human_input: str):
    client = TaskClient() # Connects to Postgres
    # Flujo rehydrates the exact state from 3 days ago and continues
    result = await client.resume_task(run_id, input_data=human_input)
    return result
```

---

## 3. Containerization Best Practices

Keep your Docker image lean. Flujo is optimized for fast cold starts.

**`Dockerfile`:**
```dockerfile
FROM python:3.13-slim

WORKDIR /app

# 1. Install system dependencies (minimal)
# Note: psutil/gcc requirements are gone in Flujo v0.7+
RUN apt-get update && rm -rf /var/lib/apt/lists/*

# 2. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy code
COPY . .

# 4. IMPORTANT: Do NOT copy the local .flujo/ folder
# Add .flujo to your .dockerignore file to prevent leaking local DBs

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**`.dockerignore`:**
```text
.flujo/
.env
__pycache__/
tests/
```

---

## 4. Observability

Production agents are hard to debug without telemetry.

1.  **Tracing:** Flujo has native OpenTelemetry hooks. Set `OTLP_ENDPOINT` (e.g., Honeycomb, Datadog, Jaeger) in your environment variables to visualize the full step execution tree.
2.  **Logging:** In production, logs should be structured JSON, not text.
    *   Set `FLUJO_LOG_JSON=1` (if your logging config supports it).
    *   Ensure your cloud provider captures `stdout`/`stderr`.

---

## 5. Security

*   **Secrets:** Never commit `.env` files. Inject API Keys (`OPENAI_API_KEY`) via your cloud provider's Secret Manager (AWS Secrets Manager / Google Secret Manager).
*   **Database Access:** Use VPC Peering or Private Service Connect to talk to Postgres. Do not expose your database to the public internet.
*   **Sandboxing:** If your agent writes code (e.g., `PythonExecStep`), **do not run it in the main container**. Use a sandboxing API (like E2B) or a separate, restricted container to execute generated code.

---

## Summary Checklist for Go-Live

- [ ] **State:** Switched `state_uri` from SQLite to Postgres.
- [ ] **Secrets:** API Keys removed from code and injected via Environment Variables.
- [ ] **Concurrency:** Configured your web server (e.g., Uvicorn) workers to match available CPU/RAM.
- [ ] **Timeout:** Verified that your platform's timeout (e.g., 60m on Cloud Run) exceeds your agent's max expected duration.
- [ ] **Telemetry:** Connected an OTel visualizer to debug complex logic chains.
