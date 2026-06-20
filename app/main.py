"""FastAPI application entrypoint + loguru configuration."""
from __future__ import annotations

import sys

from fastapi import FastAPI
from loguru import logger

from app.api.routes import ingest, query
from app.config import settings
from app.db.postgres import init_db

logger.remove()
logger.add(sys.stderr, level=settings.log_level, enqueue=True,
           backtrace=False, diagnose=False)
logger.add("logs/app.log", level=settings.log_level, rotation="50 MB",
           retention="10 days", enqueue=True)

app = FastAPI(title="RAG URL Ingestion API", version="1.0.0")
app.include_router(ingest.router)
app.include_router(query.router)


@app.on_event("startup")
async def _startup() -> None:
    await init_db()
    logger.info("Started {} ({} backend)", settings.app_name, settings.vector_backend)


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok", "backend": settings.vector_backend}
