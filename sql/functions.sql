-- xMemory Framework — Graph Recall Function
-- Expands memory recall via knowledge graph traversal

CREATE OR REPLACE FUNCTION graph_recall(
    query_embedding  vector(384),
    query_bank_id    TEXT,
    top_k            INTEGER DEFAULT 10,
    expansion_depth  INTEGER DEFAULT 1,
    min_weight       NUMERIC DEFAULT 0.3
)
RETURNS TABLE (
    id               UUID,
    text             TEXT,
    fact_type        TEXT,
    confidence_score NUMERIC,
    similarity       NUMERIC,
    depth            INTEGER,
    path_weight      NUMERIC
) AS $$
WITH RECURSIVE seed AS (
    -- Initial similarity search
    SELECT
        m.id,
        m.text,
        m.fact_type,
        m.confidence_score,
        1 - (m.embedding <=> query_embedding) AS similarity,
        0 AS depth,
        1.0 AS path_weight
    FROM memory_units m
    WHERE m.bank_id = query_bank_id
      AND m.embedding IS NOT NULL
    ORDER BY m.embedding <=> query_embedding
    LIMIT top_k
),
expanded AS (
    -- Recursive graph expansion
    SELECT * FROM seed
    UNION ALL
    SELECT
        m.id,
        m.text,
        m.fact_type,
        m.confidence_score,
        1 - (m.embedding <=> query_embedding) AS similarity,
        e.depth + 1,
        LEAST(e.path_weight, ml.weight) AS path_weight
    FROM expanded e
    JOIN memory_links ml ON (ml.source_id = e.id OR ml.target_id = e.id)
    JOIN memory_units m ON (
        CASE WHEN ml.source_id = e.id THEN ml.target_id ELSE ml.source_id END = m.id
    )
    WHERE e.depth < expansion_depth
      AND ml.weight >= min_weight
      AND m.bank_id = query_bank_id
)
SELECT DISTINCT ON (id)
    id, text, fact_type, confidence_score,
    similarity, depth, path_weight
FROM expanded
ORDER BY id, path_weight DESC, similarity DESC;
$$ LANGUAGE sql STABLE;
