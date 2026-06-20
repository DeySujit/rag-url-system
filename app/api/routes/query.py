"""Query endpoint — retrieval-augmented answering over ingested content."""
from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import QueryRequest, QueryResponse
from app.services.rag_service import RAGService

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    service = RAGService()
    return await service.query(req.query, top_k=req.top_k)
