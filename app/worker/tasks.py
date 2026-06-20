"""Celery tasks. Bridges the sync Celery worker to the async pipeline."""
from __future__ import annotations

import asyncio

from loguru import logger

from app.worker.celery_app import celery_app


@celery_app.task(
    name="ingest_url",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
)
def ingest_url_task(self, url: str, job_id: str, max_depth=None, max_pages=None) -> dict:
    """Run the async ingestion pipeline inside the sync Celery worker."""
    logger.info("Worker picked up job {} for {}", job_id, url)
    try:
        result = asyncio.run(
            _run(url, job_id, max_depth, max_pages)
        )
        return result
    except Exception as exc:  # transient infra failure -> retry with backoff
        logger.exception("Task failed for {}; retrying", url)
        raise self.retry(exc=exc)


async def _run(url: str, job_id: str, max_depth, max_pages) -> dict:
    from app.services.ingest_service import IngestionService

    service = IngestionService()
    result = await service.ingest_url(
        url, max_depth=max_depth, max_pages=max_pages, job_id=job_id
    )
    return result.model_dump(mode="json")
