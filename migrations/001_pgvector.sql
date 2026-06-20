-- pgvector bootstrap. The application also creates these on startup
-- (ensure_collection / init_db), but this file lets you provision manually.
-- Replace 2048 with EMBEDDING_DIMENSION if you change it.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
    id        TEXT PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT 'default',
    embedding vector(2048) NOT NULL,
    text      TEXT NOT NULL,
    metadata  JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS rag_chunks_emb_idx
    ON rag_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS rag_chunks_ns_idx ON rag_chunks (namespace);

-- Bookkeeping tables (also created by SQLAlchemy init_db()).
CREATE TABLE IF NOT EXISTS ingest_jobs (
    job_id          VARCHAR(64) PRIMARY KEY,
    url             TEXT NOT NULL,
    status          VARCHAR(32) DEFAULT 'pending',
    pages_crawled   INTEGER DEFAULT 0,
    chunks_total    INTEGER DEFAULT 0,
    chunks_new      INTEGER DEFAULT 0,
    chunks_skipped  INTEGER DEFAULT 0,
    chunks_upserted INTEGER DEFAULT 0,
    error           TEXT,
    started_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now(),
    finished_at     TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS indexed_chunks (
    content_hash VARCHAR(64) PRIMARY KEY,
    chunk_id     VARCHAR(128) NOT NULL,
    source_url   TEXT NOT NULL,
    job_id       VARCHAR(64),
    token_count  INTEGER DEFAULT 0,
    created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS indexed_chunks_src_idx ON indexed_chunks (source_url);
