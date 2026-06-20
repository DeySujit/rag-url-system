"""Celery application — Redis broker/result backend for background ingestion."""
from __future__ import annotations

from celery import Celery

from app.config import settings

celery_app = Celery(
    "rag",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,                 # redeliver on worker crash (recovery)
    worker_prefetch_multiplier=1,
    task_track_started=True,
    result_expires=86400,
    broker_connection_retry_on_startup=True,
)
