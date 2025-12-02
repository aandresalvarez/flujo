BEGIN;

-- Create GIN index for efficient JSONB querying on metadata
-- This index supports TaskClient.list_tasks(metadata_filter={...}) queries
-- at scale by enabling fast containment checks on the metadata JSONB column.
-- Note: metadata is stored in workflow_state table, which is joined in list_runs queries.
CREATE INDEX IF NOT EXISTS idx_workflow_state_metadata_gin ON workflow_state USING GIN (metadata);

-- Update schema version
INSERT INTO flujo_schema_versions (version, applied_at) 
VALUES (2, NOW())
ON CONFLICT (version) DO NOTHING;

COMMIT;

