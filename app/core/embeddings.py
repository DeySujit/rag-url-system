"""Async embedding generation (OpenAI) with batching, retry and rate limiting.

Uses `text-embedding-3-large`, which supports the `dimensions` parameter — we
request 2048-dim vectors to match the configured vector store. Batches are sent
concurrently up to `embedding_max_concurrency`, each call retried with
exponential backoff.
"""
from __future__ import annotations

import asyncio
from typing import Sequence

from loguru import logger
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.config import settings

try:
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    _RETRYABLE: tuple[type[Exception], ...] = (
        APIConnectionError, APITimeoutError, RateLimitError,
    )
except Exception:  # pragma: no cover
    _RETRYABLE = (Exception,)


def _chunked(items: Sequence, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


class Embedder:
    def __init__(
        self,
        model: str | None = None,
        dimension: int | None = None,
        batch_size: int | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self.model = model or settings.embedding_model
        self.dimension = dimension or settings.embedding_dimension
        self.batch_size = batch_size or settings.embedding_batch_size
        self._sem = asyncio.Semaphore(max_concurrency or settings.embedding_max_concurrency)
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.embedding_timeout_seconds,
            max_retries=0,  # we manage retries via tenacity for full control
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(settings.embedding_max_retries),
        wait=wait_random_exponential(multiplier=1, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        async with self._sem:
            resp = await self._client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimension,
            )
        # API preserves input order, but sort defensively by index.
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in ordered]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts. Returns vectors in the same order as `texts`."""
        if not texts:
            return []

        batches = list(_chunked(texts, self.batch_size))
        logger.info(
            "Embedding {} texts in {} batch(es) (model={} dim={})",
            len(texts), len(batches), self.model, self.dimension,
        )

        async def run(idx: int, batch: list[str]):
            vectors = await self._embed_batch(batch)
            logger.debug("Embedded batch {}/{} ({} vectors)", idx + 1, len(batches), len(vectors))
            return idx, vectors

        results = await asyncio.gather(*(run(i, b) for i, b in enumerate(batches)))

        ordered: list[list[float]] = []
        for _idx, vectors in sorted(results, key=lambda x: x[0]):
            ordered.extend(vectors)

        if len(ordered) != len(texts):  # pragma: no cover - defensive
            raise RuntimeError(
                f"Embedding count mismatch: got {len(ordered)} for {len(texts)} texts"
            )
        return ordered

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed_batch([text])
        return vectors[0]
