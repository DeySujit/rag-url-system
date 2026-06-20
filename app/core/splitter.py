"""Semantic, hierarchy-aware markdown chunking.

Guarantees:
  * target 800-1200 tokens/chunk, 150-200 token overlap
  * headings stay attached to their content
  * code blocks (``` fences) are never split
  * markdown tables are never split
  * document hierarchy (H1 > H2 > H3 ...) preserved as a breadcrumb `section`
  * every chunk carries metadata: chunk_id, source_url, title, section,
    token_count (+ parent_url, content_type, content_hash, chunk_index)

Strategy: split markdown into atomic "blocks" (a heading, a paragraph, a code
fence, a table, a list) while tracking the live heading stack. Greedily pack
blocks into chunks up to the token budget, never breaking an atomic block, then
add token-based overlap between consecutive chunks of the same section.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.config import settings
from app.models.schemas import Chunk, ChunkMetadata, CrawledPage

_ENCODER = tiktoken.get_encoding(settings.tokenizer_encoding)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^```")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _truncate_tokens(text: str, max_tokens: int) -> str:
    ids = _ENCODER.encode(text)
    if len(ids) <= max_tokens:
        return text
    return _ENCODER.decode(ids[:max_tokens])


def _tail_tokens(text: str, n_tokens: int) -> str:
    ids = _ENCODER.encode(text)
    if len(ids) <= n_tokens:
        return text
    return _ENCODER.decode(ids[-n_tokens:])


@dataclass
class _Block:
    text: str
    kind: str                      # heading | code | table | text
    level: int = 0                 # heading level (1-6) else 0
    headings: tuple[str, ...] = field(default_factory=tuple)  # breadcrumb
    tokens: int = 0
    atomic: bool = False           # must never be split internally


def _parse_blocks(markdown: str) -> list[_Block]:
    """Split markdown into atomic blocks while tracking the heading stack."""
    lines = markdown.splitlines()
    blocks: list[_Block] = []
    stack: list[tuple[int, str]] = []  # (level, text)
    i = 0
    n = len(lines)

    def current_breadcrumb() -> tuple[str, ...]:
        return tuple(text for _lvl, text in stack)

    while i < n:
        line = lines[i]

        # Fenced code block -> atomic
        if _FENCE_RE.match(line.strip()):
            buf = [line]
            i += 1
            while i < n and not _FENCE_RE.match(lines[i].strip()):
                buf.append(lines[i])
                i += 1
            if i < n:
                buf.append(lines[i])  # closing fence
                i += 1
            text = "\n".join(buf)
            blocks.append(
                _Block(text, "code", headings=current_breadcrumb(),
                       tokens=count_tokens(text), atomic=True)
            )
            continue

        # Markdown table -> atomic
        if _TABLE_ROW_RE.match(line):
            buf = [line]
            i += 1
            while i < n and _TABLE_ROW_RE.match(lines[i]):
                buf.append(lines[i])
                i += 1
            text = "\n".join(buf)
            blocks.append(
                _Block(text, "table", headings=current_breadcrumb(),
                       tokens=count_tokens(text), atomic=True)
            )
            continue

        # Heading -> updates the stack
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            htext = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, htext))
            blocks.append(
                _Block(line, "heading", level=level, headings=current_breadcrumb(),
                       tokens=count_tokens(line))
            )
            i += 1
            continue

        # Blank line
        if not line.strip():
            i += 1
            continue

        # Paragraph / list: accumulate until blank line or structural boundary
        buf = [line]
        i += 1
        while i < n and lines[i].strip() and not _HEADING_RE.match(lines[i]) \
                and not _FENCE_RE.match(lines[i].strip()) \
                and not _TABLE_ROW_RE.match(lines[i]):
            buf.append(lines[i])
            i += 1
        text = "\n".join(buf)
        blocks.append(
            _Block(text, "text", headings=current_breadcrumb(), tokens=count_tokens(text))
        )

    return blocks


def _split_oversized_text(block: _Block, max_tokens: int) -> list[_Block]:
    """A prose block bigger than the budget is split on sentence/word bounds."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=settings.tokenizer_encoding,
        chunk_size=max_tokens,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )
    pieces = splitter.split_text(block.text)
    return [
        _Block(p, "text", headings=block.headings, tokens=count_tokens(p))
        for p in pieces
    ]


class SemanticChunker:
    def __init__(
        self,
        target_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        overlap_tokens: Optional[int] = None,
    ) -> None:
        self.target = target_tokens or settings.chunk_target_tokens
        self.max = max_tokens or settings.chunk_max_tokens
        self.overlap = overlap_tokens or settings.chunk_overlap_tokens

    def chunk_page(self, page: CrawledPage) -> list[Chunk]:
        blocks = _parse_blocks(page.markdown)
        if not blocks:
            return []

        # Expand only oversized *prose* blocks; code/tables stay atomic.
        expanded: list[_Block] = []
        for b in blocks:
            if b.kind == "text" and b.tokens > self.max:
                expanded.extend(_split_oversized_text(b, self.max))
            else:
                expanded.append(b)

        raw_chunks = self._pack(expanded)
        raw_chunks = self._apply_overlap(raw_chunks)
        return self._finalize(raw_chunks, page)

    # --------------------------------------------------------------------- #
    def _pack(self, blocks: list[_Block]) -> list[dict]:
        """Greedily pack blocks into chunks up to the target token budget."""
        chunks: list[dict] = []
        cur_blocks: list[_Block] = []
        cur_tokens = 0

        def flush() -> None:
            nonlocal cur_blocks, cur_tokens
            if not cur_blocks:
                return
            text = "\n\n".join(b.text for b in cur_blocks).strip()
            section = " > ".join(cur_blocks[-1].headings)
            chunks.append({"text": text, "section": section,
                           "tokens": count_tokens(text)})
            cur_blocks = []
            cur_tokens = 0

        for b in blocks:
            # An atomic block larger than max gets its own chunk.
            if b.atomic and b.tokens > self.max:
                flush()
                chunks.append({
                    "text": b.text,
                    "section": " > ".join(b.headings),
                    "tokens": b.tokens,
                })
                continue

            # Starting a new heading section near budget -> flush first so the
            # heading stays with the content that follows it.
            if b.kind == "heading" and cur_tokens >= self.target:
                flush()

            if cur_tokens + b.tokens > self.max and cur_blocks:
                flush()

            cur_blocks.append(b)
            cur_tokens += b.tokens

            if cur_tokens >= self.target and not (b.kind == "heading"):
                flush()

        flush()
        return chunks

    def _apply_overlap(self, chunks: list[dict]) -> list[dict]:
        """Prepend the tail of the previous chunk for context continuity."""
        if self.overlap <= 0:
            return chunks
        out: list[dict] = []
        for idx, ch in enumerate(chunks):
            if idx == 0:
                out.append(ch)
                continue
            tail = _tail_tokens(chunks[idx - 1]["text"], self.overlap)
            merged = f"{tail}\n\n{ch['text']}".strip()
            out.append({
                "text": merged,
                "section": ch["section"],
                "tokens": count_tokens(merged),
            })
        return out

    def _finalize(self, raw_chunks: list[dict], page: CrawledPage) -> list[Chunk]:
        out: list[Chunk] = []
        for idx, ch in enumerate(raw_chunks):
            text = ch["text"].strip()
            if not text:
                continue
            content_hash = Chunk.compute_hash(page.source_url, text)
            meta = ChunkMetadata(
                chunk_id=f"{content_hash[:16]}-{idx}",
                source_url=page.source_url,
                title=page.page_title,
                section=ch["section"],
                token_count=ch["tokens"],
                parent_url=page.parent_url,
                content_type=page.content_type,
                crawl_timestamp=page.crawl_timestamp,
                chunk_index=idx,
                content_hash=content_hash,
            )
            out.append(Chunk(text=text, metadata=meta))
        logger.debug("Chunked {} into {} chunks", page.source_url, len(out))
        return out

    def chunk_pages(self, pages: list[CrawledPage]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in pages:
            chunks.extend(self.chunk_page(page))
        logger.info("Produced {} chunks from {} pages", len(chunks), len(pages))
        return chunks
