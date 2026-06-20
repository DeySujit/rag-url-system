"""Ingestion endpoint.

Submit a URL -> crawl -> chunk -> embed -> store in the vector DB.
Runs synchronously (no Celery/Redis needed): the request returns once the
URL has been fully ingested.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import IngestRequest, IngestResult
from app.services.ingest_service import ingest_url

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResult)
async def submit_ingest(req: IngestRequest) -> IngestResult:
    """Fetch the URL's content and store its embeddings in the vector DB."""
    return await ingest_url(
        str(req.url),
        max_depth=req.max_depth,
        max_pages=req.max_pages,
        from_sitemap=req.from_sitemap,
    )
