BEGIN;

CREATE TABLE IF NOT EXISTS workflow_state (
    run_id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL,
    pipeline_name TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    current_step_index INTEGER NOT NULL,
    pipeline_context JSONB,
    last_step_output JSONB,
    step_history JSONB,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    total_steps INTEGER DEFAULT 0,
    error_message TEXT,
    execution_time_ms INTEGER,
    memory_usage_mb REAL,
    metadata JSONB,
    is_background_task BOOLEAN DEFAULT FALSE,
    parent_run_id TEXT,
    task_id TEXT,
    background_error TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL,
    pipeline_name TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    execution_time_ms INTEGER,
    memory_usage_mb REAL,
    total_steps INTEGER DEFAULT 0,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    step_name TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    output JSONB,
    raw_response JSONB,
    cost_usd REAL,
    token_counts INTEGER,
    execution_time_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS traces (
    run_id TEXT PRIMARY KEY,
    trace_data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS spans (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    parent_span_id TEXT,
    name TEXT NOT NULL,
    start_time DOUBLE PRECISION NOT NULL,
    end_time DOUBLE PRECISION,
    status TEXT NOT NULL,
    attributes JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS flujo_schema_versions (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_workflow_state_status ON workflow_state(status);
CREATE INDEX IF NOT EXISTS idx_workflow_state_pipeline_id ON workflow_state(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_workflow_state_parent_run_id ON workflow_state(parent_run_id);
CREATE INDEX IF NOT EXISTS idx_workflow_state_created_at ON workflow_state(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_pipeline_id ON runs(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_pipeline_name ON runs(pipeline_name);
CREATE INDEX IF NOT EXISTS idx_steps_run_id ON steps(run_id);
CREATE INDEX IF NOT EXISTS idx_steps_step_index ON steps(step_index);
CREATE INDEX IF NOT EXISTS idx_spans_run_id ON spans(run_id);
CREATE INDEX IF NOT EXISTS idx_spans_parent_span ON spans(parent_span_id);
CREATE INDEX IF NOT EXISTS idx_runs_context_gin ON workflow_state USING GIN(pipeline_context);

COMMIT;
