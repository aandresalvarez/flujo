BEGIN;

-- Enable pgvector extension if available (safe to run if already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- Memories table for durable vector store (RAG)
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    vector vector(1536),
    payload JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_memories_embedding
ON memories
USING hnsw (vector vector_cosine_ops);

-- Track schema version
INSERT INTO flujo_schema_versions (version, applied_at)
VALUES (3, NOW())
ON CONFLICT (version) DO NOTHING;

COMMIT;

