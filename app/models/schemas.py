"""Pydantic data models shared across the ingestion pipeline."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, HttpUrl


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Crawling
# --------------------------------------------------------------------------- #
class CrawledPage(BaseModel):
    """A single fetched + cleaned web page."""

    source_url: str
    page_title: str = ""
    markdown: str = ""
    content_type: str = "text/html"
    parent_url: Optional[str] = None
    crawl_timestamp: datetime = Field(default_factory=utcnow)
    depth: int = 0
    links: list[str] = Field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.markdown.split())


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
class ChunkMetadata(BaseModel):
    chunk_id: str
    source_url: str
    title: str = ""
    section: str = ""
    token_count: int = 0
    parent_url: Optional[str] = None
    content_type: str = "text/html"
    crawl_timestamp: datetime = Field(default_factory=utcnow)
    chunk_index: int = 0
    content_hash: str = ""

    def as_payload(self) -> dict[str, Any]:
        """JSON-serialisable dict suitable for any vector DB payload."""
        data = self.model_dump()
        data["crawl_timestamp"] = self.crawl_timestamp.isoformat()
        return data


class Chunk(BaseModel):
    text: str
    metadata: ChunkMetadata

    @staticmethod
    def compute_hash(source_url: str, text: str) -> str:
        """Stable content hash used for dedup + deterministic ids.

        Scoped by source_url so identical boilerplate on different pages
        does not collide, while exact re-ingests of the same page do.
        """
        h = hashlib.sha256()
        h.update(source_url.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.strip().encode("utf-8"))
        return h.hexdigest()


# --------------------------------------------------------------------------- #
# Vector records
# --------------------------------------------------------------------------- #
class VectorRecord(BaseModel):
    id: str
    embedding: list[float]
    text: str
    metadata: dict[str, Any]


# --------------------------------------------------------------------------- #
# Ingestion status tracking
# --------------------------------------------------------------------------- #
class IngestStatus(str, Enum):
    PENDING = "pending"
    CRAWLING = "crawling"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    UPSERTING = "upserting"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestRequest(BaseModel):
    url: HttpUrl
    # depth 0 = seed page only (don't follow links); omit to use the configured default
    max_depth: Optional[int] = Field(default=None, ge=0)
    # must crawl at least 1 page; 0 would cap the crawl at zero pages
    max_pages: Optional[int] = Field(default=None, ge=1)
    # crawl every URL in the site's sitemap.xml instead of following in-page
    # links; the most reliable way to ingest an entire docs site
    from_sitemap: bool = False


class IngestResult(BaseModel):
    job_id: str
    url: str
    status: IngestStatus = IngestStatus.PENDING
    pages_crawled: int = 0
    chunks_total: int = 0
    chunks_new: int = 0
    chunks_skipped: int = 0
    chunks_upserted: int = 0
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: Optional[datetime] = None
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[RetrievedChunk] = Field(default_factory=list)
