BEGIN;

CREATE TABLE IF NOT EXISTS evaluations (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    step_name TEXT,
    score DOUBLE PRECISION NOT NULL,
    feedback TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_evaluations_run_id ON evaluations(run_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_created_at ON evaluations(created_at);

INSERT INTO flujo_schema_versions (version, applied_at)
VALUES (4, NOW())
ON CONFLICT (version) DO NOTHING;

COMMIT;

