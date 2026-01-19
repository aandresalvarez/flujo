BEGIN;

-- Drop the existing HNSW index if it exists (it may be failing due to dimensionless vectors)
DROP INDEX IF EXISTS idx_memories_embedding;

-- Attempt to alter the vector column to add dimensions
-- This works for empty tables or tables where all vectors have 1536 dimensions
-- For tables with inconsistent dimensions, manual intervention is required
DO $$
BEGIN
    -- Try to alter the column type
    BEGIN
        ALTER TABLE memories ALTER COLUMN vector TYPE vector(1536);
    EXCEPTION
        WHEN OTHERS THEN
            -- If alteration fails (e.g., data with wrong dimensions exists),
            -- log a notice. Administrators should manually clean up the data.
            RAISE NOTICE 'Could not alter vector column to vector(1536). Existing data may have incompatible dimensions.';
            RAISE NOTICE 'Please manually verify vector dimensions and clean up if necessary.';
    END;
END $$;

-- Recreate the HNSW index with proper vector dimensions
CREATE INDEX IF NOT EXISTS idx_memories_embedding
ON memories
USING hnsw (vector vector_cosine_ops);

-- Track schema version
INSERT INTO flujo_schema_versions (version, applied_at)
VALUES (5, NOW())
ON CONFLICT (version) DO NOTHING;

COMMIT;
