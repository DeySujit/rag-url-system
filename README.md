# RAG URL Ingestion System

Production web-content ingestion pipeline for Retrieval-Augmented Generation.

Given a URL it will **crawl → extract clean markdown → semantically chunk →
embed (2048-dim) → deduplicate → upsert into a vector store**, with full async
processing, rate limiting, retries, and ingestion-status tracking.

```python
from app.services.ingest_service import ingest_url
await ingest_url("https://example.com")
```

---

## Features

| Requirement            | Implementation |
|------------------------|----------------|
| Crawl + nested links   | `app/core/loader.py` — Crawl4AI primary, httpx+BeautifulSoup fallback, BFS with depth/page limits, same-domain filter |
| Boilerplate removal    | strips `script/nav/header/footer/aside/ads/...`, prefers `<main>`/`<article>` |
| Clean markdown         | hierarchy-preserving emitter (headings, code fences, tables, lists) |
| Metadata               | `source_url, page_title, crawl_timestamp, content_type, parent_url` |
| Semantic chunking      | `app/core/splitter.py` — 800–1200 tokens, 150–200 overlap, never splits code/tables, keeps headings, section breadcrumbs |
| Embeddings             | `app/core/embeddings.py` — OpenAI `text-embedding-3-large` @ dim 2048, batched, async, retry |
| Vector DB abstraction  | `app/core/vectorstore.py` — pgvector / Pinecone / Qdrant / Weaviate / Milvus behind one interface |
| Dedup + skip indexed   | content-hash ids + `indexed_chunks` table (`app/db/metadata.py`) |
| Status tracking        | `ingest_jobs` table, per-stage updates, partial-failure recovery |
| Async + scale          | asyncio crawling/embedding, bounded concurrency, batched upserts |

---

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# For the primary crawler (Crawl4AI) — optional, falls back to httpx if absent:
python -m playwright install --with-deps chromium

# Install ONLY the vector backend you use (pgvector needs nothing extra):
# pip install pinecone-client qdrant-client weaviate-client pymilvus
```

## 2. Configure

```bash
cp .env.example .env      # then fill in OPENAI_API_KEY and DB settings
```

Key variables: `OPENAI_API_KEY`, `VECTOR_BACKEND` (default `pgvector`),
`DB_*_LOCAL`. See `.env.example` for the full list.

## 3. Postgres (pgvector)

Use a local PostgreSQL with the `pgvector` extension, reachable at the
`DB_*_LOCAL` settings above. Enable the extension once per database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The app auto-creates its tables and the vector collection on first run.

## 4. Ingest

```bash
uvicorn app.main:app --reload                                  # terminal 1

curl -X POST localhost:8000/ingest -H 'content-type: application/json' \
     -d '{"url":"https://example.com","max_depth":1,"max_pages":20}'
# -> runs synchronously, returns the IngestResult when done

curl -X POST localhost:8000/query -H 'content-type: application/json' \
     -d '{"query":"What is this site about?","top_k":5}'
```

## 5. Test

```bash
pytest -q          # offline unit tests (no network / DB / OpenAI needed)
```

---

## Project structure

```
app/
├── main.py                  FastAPI app + logging
├── config.py                pydantic-settings (all env vars)
├── api/routes/
│   ├── ingest.py            POST /ingest -> crawl+embed+store (synchronous)
│   └── query.py             POST /query  -> RAG answer
├── core/
│   ├── loader.py            async crawler + HTML->markdown cleaning
│   ├── splitter.py          semantic, hierarchy-aware chunker
│   ├── embeddings.py        async batched OpenAI embeddings (dim 2048)
│   ├── vectorstore.py       5-backend abstraction + factory
│   ├── llm.py               chat wrapper
│   └── prompt.py            RAG prompt templates
├── db/
│   ├── postgres.py          SQLAlchemy async engine + models + init_db
│   └── metadata.py          dedup + status-tracking helpers
├── services/
│   ├── ingest_service.py    ingest_url() orchestration
│   └── rag_service.py       retrieval + answer generation
└── models/schemas.py        pydantic models (Chunk, VectorRecord, ...)
examples/ingest_example.py   runnable end-to-end demo
migrations/001_pgvector.sql  manual schema bootstrap
tests/                       offline unit tests
```

## Switching vector backends

Set `VECTOR_BACKEND` to `pgvector | pinecone | qdrant | weaviate | milvus`,
install that backend's client, and fill its `*_API_KEY/URL` env vars. No code
changes — `get_vector_store()` returns the right implementation and every record
keeps the uniform `{id, embedding, text, metadata}` shape.

## Scaling notes

- **Idempotent**: vector ids are content hashes, so re-ingesting a URL upserts
  in place and `filter_new_chunks()` skips anything already in `indexed_chunks`.
- **Throughput**: tune `CRAWL_CONCURRENCY`, `EMBEDDING_MAX_CONCURRENCY`,
  `EMBEDDING_BATCH_SIZE`, `UPSERT_BATCH_SIZE`.
- **Recovery**: each stage persists status to `ingest_jobs`, so a failed run
  records exactly where it stopped.
