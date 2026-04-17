-- xMemory Framework — Database Schema
-- Compatible with PostgreSQL 14+ and pgvector 0.5+

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─── Banks ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS banks (
    bank_id     TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    mission     TEXT DEFAULT '',
    disposition JSONB DEFAULT '{"skepticism": 3, "literalism": 3, "empathy": 3}',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Memory Units ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memory_units (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    bank_id           TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    text              TEXT NOT NULL,
    context           TEXT DEFAULT '',
    embedding         vector(384),
    fact_type         TEXT DEFAULT 'observation' CHECK (fact_type IN ('world', 'experience', 'observation')),
    confidence_score  NUMERIC(3,2) DEFAULT NULL,
    access_count      INTEGER DEFAULT 0,
    tags              VARCHAR(64)[] DEFAULT '{}',
    mentioned_at      TIMESTAMPTZ DEFAULT NOW(),
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    consolidated_at   TIMESTAMPTZ,
    metadata          JSONB DEFAULT '{}',
    history           JSONB DEFAULT '[]',
    observation_scopes JSONB DEFAULT '[]',
    text_signals      TEXT,
    source_memory_ids UUID[] DEFAULT '{}',
    proof_count       INTEGER DEFAULT 1
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_memory_bank ON memory_units(bank_id);
CREATE INDEX IF NOT EXISTS idx_memory_fact_type ON memory_units(bank_id, fact_type);
CREATE INDEX IF NOT EXISTS idx_memory_created ON memory_units(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_confidence ON memory_units(confidence_score) WHERE confidence_score IS NOT NULL;

-- HNSW index for vector similarity (fast recall)
CREATE INDEX IF NOT EXISTS idx_memory_embedding ON memory_units
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ─── Memory Links (Knowledge Graph) ──────────────────────────────────
CREATE TABLE IF NOT EXISTS memory_links (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id   UUID NOT NULL REFERENCES memory_units(id) ON DELETE CASCADE,
    target_id   UUID NOT NULL REFERENCES memory_units(id) ON DELETE CASCADE,
    link_type   TEXT NOT NULL CHECK (link_type IN ('entity', 'temporal', 'semantic', 'caused_by', 'cross_bank')),
    weight      NUMERIC(5,4) DEFAULT 1.0,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_id, target_id, link_type)
);

CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_id);
CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_id);
CREATE INDEX IF NOT EXISTS idx_links_type ON memory_links(link_type);

-- ─── Async Operations ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS async_operations (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    bank_id     TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    operation   TEXT NOT NULL,
    status      TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    result      JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Helper Functions ────────────────────────────────────────────────

-- Update bank timestamp on memory changes
CREATE OR REPLACE FUNCTION update_bank_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE banks SET updated_at = NOW() WHERE bank_id = COALESCE(NEW.bank_id, OLD.bank_id);
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_update_bank ON memory_units;
CREATE TRIGGER trg_update_bank
    AFTER INSERT OR UPDATE OR DELETE ON memory_units
    FOR EACH ROW EXECUTE FUNCTION update_bank_timestamp();
