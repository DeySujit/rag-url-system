"""Prompt templates for the RAG answer-generation step."""
from __future__ import annotations

from app.models.schemas import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the user's question using ONLY the "
    "provided context. If the context is insufficient, say so plainly. Cite "
    "sources inline as [n] referring to the numbered context blocks."
)


def build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        src = c.metadata.get("source_url", "")
        section = c.metadata.get("section", "")
        header = f"[{i}] {src}" + (f" — {section}" if section else "")
        blocks.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = build_context(chunks)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer (cite sources as [n]):"
    )
