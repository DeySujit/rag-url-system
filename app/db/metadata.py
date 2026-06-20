"""Helpers for ingestion-status tracking and content-hash deduplication."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.postgres import IndexedChunk, IngestJob, get_sessionmaker
from app.models.schemas import Chunk, IngestResult, IngestStatus


async def create_job(job_id: str, url: str) -> None:
    async with get_sessionmaker()() as s:
        s.add(IngestJob(job_id=job_id, url=url, status=IngestStatus.PENDING.value))
        await s.commit()


async def update_job(job_id: str, **fields) -> None:
    if "status" in fields and isinstance(fields["status"], IngestStatus):
        fields["status"] = fields["status"].value
    async with get_sessionmaker()() as s:
        await s.execute(update(IngestJob).where(IngestJob.job_id == job_id).values(**fields))
        await s.commit()


async def finish_job(result: IngestResult) -> None:
    await update_job(
        result.job_id,
        status=result.status,
        pages_crawled=result.pages_crawled,
        chunks_total=result.chunks_total,
        chunks_new=result.chunks_new,
        chunks_skipped=result.chunks_skipped,
        chunks_upserted=result.chunks_upserted,
        error=result.error,
        finished_at=datetime.now(timezone.utc),
    )


async def get_job(job_id: str) -> IngestJob | None:
    async with get_sessionmaker()() as s:
        return await s.get(IngestJob, job_id)


async def filter_new_chunks(chunks: list[Chunk]) -> tuple[list[Chunk], int]:
    """Return (chunks not yet indexed, skipped_count).

    Deduplicates within the batch and against `indexed_chunks` in Postgres.
    """
    if not chunks:
        return [], 0

    # In-batch dedup first.
    seen: set[str] = set()
    deduped: list[Chunk] = []
    for c in chunks:
        h = c.metadata.content_hash
        if h in seen:
            continue
        seen.add(h)
        deduped.append(c)

    hashes = [c.metadata.content_hash for c in deduped]
    async with get_sessionmaker()() as s:
        rows = await s.execute(
            select(IndexedChunk.content_hash).where(IndexedChunk.content_hash.in_(hashes))
        )
        existing = {r[0] for r in rows.all()}

    new_chunks = [c for c in deduped if c.metadata.content_hash not in existing]
    skipped = len(chunks) - len(new_chunks)
    logger.info(
        "Dedup: {} total -> {} new, {} skipped (already indexed/duplicate)",
        len(chunks), len(new_chunks), skipped,
    )
    return new_chunks, skipped


async def mark_indexed(job_id: str, chunks: Iterable[Chunk]) -> None:
    """Record successfully upserted chunks so future runs skip them."""
    rows = [
        {
            "content_hash": c.metadata.content_hash,
            "chunk_id": c.metadata.chunk_id,
            "source_url": c.metadata.source_url,
            "job_id": job_id,
            "token_count": c.metadata.token_count,
        }
        for c in chunks
    ]
    if not rows:
        return
    async with get_sessionmaker()() as s:
        stmt = pg_insert(IndexedChunk).values(rows)
        stmt = stmt.on_conflict_do_nothing(index_elements=["content_hash"])
        await s.execute(stmt)
        await s.commit()
