"""Pluggable async embedding generation.

A single `BaseEmbedder` interface with interchangeable backends:
    openai | gemini | voyage | cohere

Select via `settings.embedding_backend`. Provider SDKs are imported lazily so you
only need the dependency for the backend you actually use.

  ⚠️  Claude/Anthropic and Groq are NOT embedding providers — their APIs are
      chat-only (no `/embeddings` endpoint). Selecting them raises a clear error.

The base class owns the shared machinery — batching, bounded-concurrency
fan-out, ordering, and retry with exponential backoff. Each backend only
implements `_embed_raw()`: turn a batch of strings into a list of vectors.

`embed_texts()` embeds documents (for ingestion); `embed_query()` embeds a single
search query. Providers that distinguish the two (Gemini/Voyage/Cohere) get the
right task/input type automatically; OpenAI ignores it.
"""
from __future__ import annotations

import abc
import asyncio
from typing import Literal, Sequence

from loguru import logger
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.config import settings

InputType = Literal["document", "query"]


def _chunked(items: Sequence, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
class BaseEmbedder(abc.ABC):
    """Backend-agnostic embedder. Subclasses implement `_embed_raw()`."""

    # Exceptions worth retrying. Subclasses narrow this to the SDK's transient
    # errors so we don't retry, e.g., auth failures.
    _RETRYABLE: tuple[type[Exception], ...] = (Exception,)

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

    @abc.abstractmethod
    async def _embed_raw(self, batch: list[str], input_type: InputType) -> list[list[float]]:
        """Provider call: embed one batch, return vectors in input order."""

    async def _embed_batch(self, batch: list[str], input_type: InputType) -> list[list[float]]:
        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(settings.embedding_max_retries),
            wait=wait_random_exponential(multiplier=1, max=30),
            retry=retry_if_exception_type(self._RETRYABLE),
        ):
            with attempt:
                async with self._sem:
                    return await self._embed_raw(batch, input_type)
        raise RuntimeError("unreachable")  # pragma: no cover

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed many documents. Returns vectors in the same order as `texts`."""
        if not texts:
            return []

        batches = list(_chunked(texts, self.batch_size))
        logger.info(
            "Embedding {} texts in {} batch(es) (backend={} model={} dim={})",
            len(texts), len(batches), settings.embedding_backend, self.model, self.dimension,
        )

        async def run(idx: int, batch: list[str]):
            vectors = await self._embed_batch(batch, "document")
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
        vectors = await self._embed_batch([text], "query")
        return vectors[0]


# --------------------------------------------------------------------------- #
# OpenAI  (text-embedding-3-large / -small)
# --------------------------------------------------------------------------- #
class OpenAIEmbedder(BaseEmbedder):
    DEFAULT_MODEL = "text-embedding-3-large"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        from openai import AsyncOpenAI  # lazy

        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.embedding_timeout_seconds,
            max_retries=0,  # retries handled by the base class
        )
        try:
            from openai import APIConnectionError, APITimeoutError, RateLimitError

            self._RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError)
        except Exception:  # pragma: no cover
            pass

    async def _embed_raw(self, batch: list[str], input_type: InputType) -> list[list[float]]:
        # text-embedding-3-* honour the `dimensions` param; OpenAI has no query/doc split.
        resp = await self._client.embeddings.create(
            model=self.model, input=batch, dimensions=self.dimension
        )
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in ordered]


# --------------------------------------------------------------------------- #
# Gemini / Google  (gemini-embedding-001)
# --------------------------------------------------------------------------- #
class GeminiEmbedder(BaseEmbedder):
    DEFAULT_MODEL = "gemini-embedding-001"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        from google import genai  # lazy ; pip install google-genai

        self._client = genai.Client(api_key=settings.gemini_api_key)

    async def _embed_raw(self, batch: list[str], input_type: InputType) -> list[list[float]]:
        from google.genai import types

        task = "RETRIEVAL_QUERY" if input_type == "query" else "RETRIEVAL_DOCUMENT"
        resp = await self._client.aio.models.embed_content(
            model=self.model,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task, output_dimensionality=self.dimension
            ),
        )
        return [list(e.values) for e in resp.embeddings]


# --------------------------------------------------------------------------- #
# Voyage AI  (voyage-3-large) — Anthropic's recommended embeddings provider
# --------------------------------------------------------------------------- #
class VoyageEmbedder(BaseEmbedder):
    DEFAULT_MODEL = "voyage-3-large"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        import voyageai  # lazy ; pip install voyageai

        self._client = voyageai.AsyncClient(api_key=settings.voyage_api_key)

    async def _embed_raw(self, batch: list[str], input_type: InputType) -> list[list[float]]:
        v_type = "query" if input_type == "query" else "document"
        resp = await self._client.embed(
            batch, model=self.model, input_type=v_type, output_dimension=self.dimension
        )
        return resp.embeddings


# --------------------------------------------------------------------------- #
# Cohere  (embed-v4.0)
# --------------------------------------------------------------------------- #
class CohereEmbedder(BaseEmbedder):
    DEFAULT_MODEL = "embed-v4.0"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        import cohere  # lazy ; pip install cohere

        self._client = cohere.AsyncClientV2(api_key=settings.cohere_api_key)

    async def _embed_raw(self, batch: list[str], input_type: InputType) -> list[list[float]]:
        c_type = "search_query" if input_type == "query" else "search_document"
        resp = await self._client.embed(
            texts=batch,
            model=self.model,
            input_type=c_type,
            embedding_types=["float"],
            output_dimension=self.dimension,
        )
        return [list(v) for v in resp.embeddings.float_]


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
_EMBEDDERS: dict[str, type[BaseEmbedder]] = {
    "openai": OpenAIEmbedder,
    "gemini": GeminiEmbedder,
    "voyage": VoyageEmbedder,
    "cohere": CohereEmbedder,
}

# Chat-only providers people reach for by mistake — fail loudly with the reason.
_NO_EMBEDDINGS_API: dict[str, str] = {
    "anthropic": "Anthropic/Claude has no embeddings API (chat-only). "
                 "Anthropic recommends Voyage AI — set EMBEDDING_BACKEND=voyage.",
    "claude": "Anthropic/Claude has no embeddings API (chat-only). "
              "Anthropic recommends Voyage AI — set EMBEDDING_BACKEND=voyage.",
    "groq": "Groq has no embeddings API (chat/completions only). "
            "Use EMBEDDING_BACKEND=openai | gemini | voyage | cohere.",
}


def get_embedder(
    backend: str | None = None,
    model: str | None = None,
    dimension: int | None = None,
    batch_size: int | None = None,
    max_concurrency: int | None = None,
) -> BaseEmbedder:
    backend = (backend or settings.embedding_backend).lower()

    if backend in _NO_EMBEDDINGS_API:
        raise ValueError(_NO_EMBEDDINGS_API[backend])
    if backend not in _EMBEDDERS:
        raise ValueError(
            f"Unknown embedding backend {backend!r}. Choose one of {list(_EMBEDDERS)}"
        )

    cls = _EMBEDDERS[backend]

    # Footgun guard: if the model is still the OpenAI default but the backend
    # isn't OpenAI, fall back to that provider's default model.
    resolved_model = model or settings.embedding_model
    if backend != "openai" and resolved_model == OpenAIEmbedder.DEFAULT_MODEL:
        resolved_model = cls.DEFAULT_MODEL
        logger.warning(
            "EMBEDDING_MODEL is the OpenAI default but backend is {!r}; "
            "using {!r} instead. Set EMBEDDING_MODEL explicitly to silence this.",
            backend, resolved_model,
        )

    logger.info("Using embedding backend: {}", backend)
    return cls(
        model=resolved_model,
        dimension=dimension,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
    )
