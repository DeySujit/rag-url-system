"""Minimal end-to-end example.

    python -m examples.ingest_example https://example.com

Runs the async pipeline in-process (no Celery/Redis needed) and then issues a
RAG query against what was just ingested.
"""
from __future__ import annotations

import asyncio
import sys

from app.services.ingest_service import ingest_url
from app.services.rag_service import RAGService


async def main(url: str) -> None:
    # 1. Ingest a URL (crawl -> chunk -> embed -> upsert)
    result = await ingest_url(url, max_depth=1, max_pages=10)
    print("\n=== Ingestion result ===")
    print(result.model_dump_json(indent=2))

    if result.chunks_upserted == 0:
        return

    # 2. Ask a question over the freshly ingested content
    rag = RAGService()
    answer = await rag.query("What is this page about?", top_k=4)
    print("\n=== Answer ===")
    print(answer.answer)
    print("\n=== Sources ===")
    for s in answer.sources:
        print(f"- ({s.score:.3f}) {s.metadata.get('source_url')}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    asyncio.run(main(target))
