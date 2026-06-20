"""Retrieval-augmented answer generation over the ingested corpus."""
from __future__ import annotations

from loguru import logger

from app.core.embeddings import Embedder
from app.core.llm import LLM
from app.core.prompt import SYSTEM_PROMPT, build_user_prompt
from app.core.vectorstore import VectorStore, get_vector_store
from app.models.schemas import QueryResponse


class RAGService:
    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.store = store or get_vector_store()
        self.llm = llm or LLM()

    async def query(self, question: str, top_k: int = 5) -> QueryResponse:
        await self.store.ensure_collection()
        vector = await self.embedder.embed_query(question)
        sources = await self.store.query(vector, top_k=top_k)
        logger.info("Retrieved {} chunks for query", len(sources))
        if not sources:
            return QueryResponse(query=question, answer="No indexed content found.", sources=[])
        prompt = build_user_prompt(question, sources)
        answer = await self.llm.complete(SYSTEM_PROMPT, prompt)
        return QueryResponse(query=question, answer=answer, sources=sources)
