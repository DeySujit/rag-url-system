"""End-to-end ingestion orchestration.

    ingest_url("https://example.com")

Pipeline:  crawl -> chunk -> dedup -> embed -> upsert -> track status

Designed for scale (millions of docs):
  * async crawling + async embedding with bounded concurrency
  * content-hash dedup so re-ingests are cheap and idempotent
  * batched upserts
  * per-stage status persisted to Postgres for partial-recovery visibility
"""
from __future__ import annotations

import uuid

from loguru import logger

from app.core.embeddings import BaseEmbedder, get_embedder
from app.core.loader import WebCrawler
from app.core.splitter import SemanticChunker
from app.core.vectorstore import VectorStore, get_vector_store
from app.db import metadata as meta
from app.db.postgres import init_db
from app.models.schemas import (
    Chunk,
    IngestResult,
    IngestStatus,
    VectorRecord,
)


class IngestionService:
    def __init__(
        self,
        crawler: WebCrawler | None = None,
        chunker: SemanticChunker | None = None,
        embedder: BaseEmbedder | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.crawler = crawler or WebCrawler()
        self.chunker = chunker or SemanticChunker()
        self.embedder = embedder or get_embedder()
        self.store = store or get_vector_store()

    async def ingest_url(
        self,
        url: str,
        max_depth: int | None = None,
        max_pages: int | None = None,
        from_sitemap: bool = False,
        job_id: str | None = None,
    ) -> IngestResult:
        job_id = job_id or uuid.uuid4().hex
        result = IngestResult(job_id=job_id, url=str(url))

        await init_db()
        await meta.create_job(job_id, str(url))

        if max_depth is not None:
            self.crawler.max_depth = max_depth
        if max_pages is not None:
            self.crawler.max_pages = max_pages

        try:
            await self.store.ensure_collection()

            # 1. Crawl --------------------------------------------------------
            await meta.update_job(job_id, status=IngestStatus.CRAWLING)
            if from_sitemap:
                # Crawl every URL listed in the site's sitemap. max_pages, when
                # given, caps how many we take; otherwise we take them all.
                sitemap_urls = await self.crawler.fetch_sitemap_urls(str(url))
                if max_pages is not None:
                    sitemap_urls = sitemap_urls[:max_pages]
                pages = await self.crawler.crawl_urls(sitemap_urls)
            else:
                pages = await self.crawler.crawl(str(url))
            result.pages_crawled = len(pages)
            if not pages:
                logger.warning("No content crawled from {}", url)
                result.status = IngestStatus.COMPLETED
                await meta.finish_job(result)
                return result

            # 2. Chunk --------------------------------------------------------
            await meta.update_job(job_id, status=IngestStatus.CHUNKING,
                                  pages_crawled=result.pages_crawled)
            chunks = self.chunker.chunk_pages(pages)
            result.chunks_total = len(chunks)

            # 3. Dedup / skip already-indexed --------------------------------
            new_chunks, skipped = await meta.filter_new_chunks(chunks)
            result.chunks_skipped = skipped
            result.chunks_new = len(new_chunks)
            if not new_chunks:
                logger.info("Nothing new to index for {}", url)
                result.status = IngestStatus.COMPLETED
                await meta.finish_job(result)
                return result

            # 4. Embed --------------------------------------------------------
            await meta.update_job(job_id, status=IngestStatus.EMBEDDING,
                                  chunks_total=result.chunks_total,
                                  chunks_new=result.chunks_new,
                                  chunks_skipped=result.chunks_skipped)
            vectors = await self.embedder.embed_texts([c.text for c in new_chunks])

            # 5. Upsert -------------------------------------------------------
            await meta.update_job(job_id, status=IngestStatus.UPSERTING)
            records = [self._to_record(c, v) for c, v in zip(new_chunks, vectors)]
            upserted = await self.store.upsert(records)
            result.chunks_upserted = upserted

            # 6. Mark indexed (so subsequent runs skip these) -----------------
            await meta.mark_indexed(job_id, new_chunks)

            result.status = IngestStatus.COMPLETED
            await meta.finish_job(result)
            logger.success(
                "Ingest complete {}: {} pages, {} new chunks upserted (skipped {})",
                url, result.pages_crawled, result.chunks_upserted, result.chunks_skipped,
            )
            return result

        except Exception as exc:  # partial-ingestion recovery: persist failure
            logger.exception("Ingestion failed for {}", url)
            result.status = IngestStatus.FAILED
            result.error = f"{type(exc).__name__}: {exc}"
            await meta.finish_job(result)
            return result
        finally:
            await self.store.close()

    @staticmethod
    def _to_record(chunk: Chunk, embedding: list[float]) -> VectorRecord:
        return VectorRecord(
            id=chunk.metadata.content_hash,   # content-hash id => idempotent upsert
            embedding=embedding,
            text=chunk.text,
            metadata=chunk.metadata.as_payload(),
        )


async def ingest_url(
    url: str,
    max_depth: int | None = None,
    max_pages: int | None = None,
    from_sitemap: bool = False,
) -> IngestResult:
    """Convenience entrypoint:  await ingest_url("https://example.com")."""
    service = IngestionService()
    return await service.ingest_url(
        url, max_depth=max_depth, max_pages=max_pages, from_sitemap=from_sitemap
    )
