"""Ingestion endpoints. Submitting a URL returns a job_id immediately."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from app.db import metadata as meta
from app.models.schemas import IngestRequest, IngestResult, IngestStatus

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResult, status_code=202)
async def submit_ingest(req: IngestRequest) -> IngestResult:
    """Queue an ingestion job via Celery and return its job_id."""
    job_id = uuid.uuid4().hex
    await meta.create_job(job_id, str(req.url))

    try:
        from app.worker.tasks import ingest_url_task

        ingest_url_task.delay(
            str(req.url), job_id, req.max_depth, req.max_pages
        )
    except Exception as exc:  # broker unavailable -> surface clearly
        await meta.update_job(job_id, status=IngestStatus.FAILED, error=str(exc))
        raise HTTPException(503, f"Task queue unavailable: {exc}") from exc

    return IngestResult(job_id=job_id, url=str(req.url), status=IngestStatus.PENDING)


@router.get("/{job_id}", response_model=IngestResult)
async def get_status(job_id: str) -> IngestResult:
    job = await meta.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return IngestResult(
        job_id=job.job_id,
        url=job.url,
        status=IngestStatus(job.status),
        pages_crawled=job.pages_crawled,
        chunks_total=job.chunks_total,
        chunks_new=job.chunks_new,
        chunks_skipped=job.chunks_skipped,
        chunks_upserted=job.chunks_upserted,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
    )
