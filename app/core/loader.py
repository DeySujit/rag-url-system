"""Async web crawler + HTML→markdown content extraction.

Primary engine is Crawl4AI (headless browser, robust JS rendering and
markdown generation). If Crawl4AI is unavailable we transparently fall back
to an httpx + BeautifulSoup pipeline so the module is always importable and
testable.

Responsibilities:
  * fetch pages (async, rate limited, with timeout + retry)
  * strip nav/header/footer/ads/scripts/boilerplate
  * convert to clean markdown
  * follow nested links up to a configured depth (same-domain by default)
  * attach metadata (source_url, page_title, content_type, parent_url, ...)
"""
from __future__ import annotations

import asyncio
from collections import deque
from typing import Iterable, Optional
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.models.schemas import CrawledPage

try:  # optional heavy dependency
    from crawl4ai import AsyncWebCrawler  # type: ignore

    _HAS_CRAWL4AI = True
except Exception:  # pragma: no cover - import guard
    _HAS_CRAWL4AI = False


# Tags that are virtually always boilerplate.
_BOILERPLATE_TAGS = (
    "script", "style", "noscript", "nav", "header", "footer", "aside",
    "form", "button", "svg", "iframe", "template",
)
_BOILERPLATE_ROLES = ("navigation", "banner", "contentinfo", "search", "complementary")
_BOILERPLATE_HINTS = (
    "nav", "navbar", "menu", "sidebar", "footer", "header", "advert", "ads",
    "ad-", "promo", "cookie", "consent", "subscribe", "newsletter", "social",
    "breadcrumb", "pagination", "related", "share", "comment",
)


def normalize_url(url: str) -> str:
    """Drop fragments and trailing slashes for stable identity/dedup."""
    url, _ = urldefrag(url)
    return url.rstrip("/") or url


def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def same_domain(a: str, b: str) -> bool:
    return urlparse(a).netloc == urlparse(b).netloc


def _strip_boilerplate(soup: BeautifulSoup) -> None:
    for tag in soup(list(_BOILERPLATE_TAGS)):
        tag.decompose()

    for el in soup.find_all(attrs={"role": True}):
        if str(el.get("role", "")).lower() in _BOILERPLATE_ROLES:
            el.decompose()

    for el in soup.find_all(attrs={"class": True}) + soup.find_all(attrs={"id": True}):
        classes = el.get("class", [])
        if isinstance(classes, str):
            classes = [classes]
        ident = " ".join(filter(None, [" ".join(classes), el.get("id", "")])).lower()
        if any(hint in ident for hint in _BOILERPLATE_HINTS):
            el.decompose()


def _pick_main(soup: BeautifulSoup) -> BeautifulSoup:
    """Prefer <main>/<article> when present (documentation & blogs)."""
    main = soup.find("main") or soup.find("article")
    if main is not None:
        return BeautifulSoup(str(main), "html.parser")
    return soup


def _table_to_markdown(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def html_to_markdown(html: str, base_url: str) -> tuple[str, str, list[str]]:
    """Return (title, markdown, links) from raw HTML.

    The markdown emitter preserves headings (document hierarchy), fenced code
    blocks and tables — the structures the chunker must never split.
    """
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = normalize_url(urljoin(base_url, a["href"]))
        if href.startswith(("http://", "https://")):
            links.append(href)

    _strip_boilerplate(soup)
    content = _pick_main(soup)

    md_lines: list[str] = []

    def render(node) -> None:
        if isinstance(node, NavigableString):
            return
        if not isinstance(node, Tag):
            return

        name = node.name
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(name[1])
            md_lines.append(f"\n{'#' * level} {node.get_text(' ', strip=True)}\n")
            return
        if name == "pre":
            code = node.get_text("\n").rstrip()
            md_lines.append(f"\n```\n{code}\n```\n")
            return
        if name == "table":
            md_lines.append("\n" + _table_to_markdown(node) + "\n")
            return
        if name in ("ul", "ol"):
            ordered = name == "ol"
            for i, li in enumerate(node.find_all("li", recursive=False), start=1):
                prefix = f"{i}." if ordered else "-"
                md_lines.append(f"{prefix} {li.get_text(' ', strip=True)}")
            md_lines.append("")
            return
        if name in ("p", "blockquote"):
            text = node.get_text(" ", strip=True)
            if text:
                md_lines.append(f"\n{text}\n")
            return
        for child in node.children:
            render(child)

    body = content.body or content
    for child in body.children:
        render(child)

    markdown = "\n".join(line.rstrip() for line in md_lines)
    while "\n\n\n" in markdown:
        markdown = markdown.replace("\n\n\n", "\n\n")
    return title, markdown.strip(), links


class WebCrawler:
    """Breadth-first async crawler with depth + page limits and rate limiting."""

    def __init__(
        self,
        max_depth: Optional[int] = None,
        max_pages: Optional[int] = None,
        concurrency: Optional[int] = None,
        same_domain_only: Optional[bool] = None,
    ) -> None:
        self.max_depth = settings.crawl_max_depth if max_depth is None else max_depth
        self.max_pages = settings.crawl_max_pages if max_pages is None else max_pages
        self.same_domain_only = (
            settings.crawl_same_domain_only
            if same_domain_only is None
            else same_domain_only
        )
        self._sem = asyncio.Semaphore(concurrency or settings.crawl_concurrency)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError)),
    )
    async def _fetch_html(self, client: httpx.AsyncClient, url: str) -> tuple[str, str]:
        async with self._sem:
            resp = await client.get(url)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "text/html").split(";")[0].strip()
            return resp.text, ctype

    async def _fetch_page(
        self, client: httpx.AsyncClient, url: str, depth: int, parent: Optional[str]
    ) -> Optional[CrawledPage]:
        try:
            if _HAS_CRAWL4AI:
                page = await self._fetch_with_crawl4ai(url, depth, parent)
                if page is not None:
                    return page
            html, ctype = await self._fetch_html(client, url)
        except Exception as exc:  # invalid URL, timeout, 4xx/5xx after retries
            logger.warning("Failed to fetch {}: {}", url, exc)
            return None

        if "html" not in ctype:
            logger.debug("Skipping non-HTML content {} ({})", url, ctype)
            return None

        title, markdown, links = html_to_markdown(html, url)
        page = CrawledPage(
            source_url=normalize_url(url),
            page_title=title,
            markdown=markdown,
            content_type=ctype,
            parent_url=parent,
            depth=depth,
            links=links,
        )
        if page.word_count < settings.crawl_min_words:
            logger.debug("Dropping near-empty page {} ({} words)", url, page.word_count)
            return None
        return page

    async def _fetch_with_crawl4ai(
        self, url: str, depth: int, parent: Optional[str]
    ) -> Optional[CrawledPage]:  # pragma: no cover - requires a browser runtime
        try:
            async with AsyncWebCrawler(verbose=False) as crawler:  # type: ignore
                result = await crawler.arun(
                    url=url,
                    word_count_threshold=settings.crawl_min_words,
                    excluded_tags=list(_BOILERPLATE_TAGS),
                    exclude_external_links=self.same_domain_only,
                    page_timeout=settings.crawl_timeout_seconds * 1000,
                )
            if not result or not getattr(result, "success", False):
                return None
            markdown = (
                getattr(result, "fit_markdown", None)
                or getattr(result, "markdown", "")
                or ""
            )
            links: list[str] = []
            internal = getattr(result, "links", {}) or {}
            for group in internal.values():
                for link in group:
                    href = link.get("href") if isinstance(link, dict) else link
                    if href:
                        links.append(normalize_url(href))
            meta = getattr(result, "metadata", {}) or {}
            return CrawledPage(
                source_url=normalize_url(url),
                page_title=meta.get("title", ""),
                markdown=markdown,
                content_type="text/html",
                parent_url=parent,
                depth=depth,
                links=links,
            )
        except Exception as exc:
            logger.debug("crawl4ai failed for {} ({}); falling back", url, exc)
            return None

    async def crawl(self, seed_url: str) -> list[CrawledPage]:
        """BFS crawl starting from `seed_url`."""
        if not is_valid_url(seed_url):
            raise ValueError(f"Invalid URL: {seed_url!r}")

        seed = normalize_url(seed_url)
        seen: set[str] = {seed}
        queue: deque[tuple[str, int, Optional[str]]] = deque([(seed, 0, None)])
        pages: list[CrawledPage] = []

        headers = {"User-Agent": settings.crawl_user_agent}
        timeout = httpx.Timeout(settings.crawl_timeout_seconds)
        async with httpx.AsyncClient(
            headers=headers, timeout=timeout, follow_redirects=True
        ) as client:
            while queue and len(pages) < self.max_pages:
                level: list[tuple[str, int, Optional[str]]] = []
                while queue and len(level) + len(pages) < self.max_pages:
                    level.append(queue.popleft())

                results = await asyncio.gather(
                    *(self._fetch_page(client, u, d, p) for (u, d, p) in level)
                )

                for (url, depth, _parent), page in zip(level, results):
                    if page is None:
                        continue
                    pages.append(page)
                    logger.info(
                        "Crawled [{}/{}] depth={} {}",
                        len(pages), self.max_pages, depth, url,
                    )
                    if depth < self.max_depth:
                        for link in self._next_links(page.links, seed):
                            if link not in seen:
                                seen.add(link)
                                queue.append((link, depth + 1, url))

        logger.info("Crawl complete: {} pages from {}", len(pages), seed)
        return pages

    def _next_links(self, links: Iterable[str], seed: str) -> list[str]:
        out = []
        for link in links:
            if self.same_domain_only and not same_domain(link, seed):
                continue
            out.append(normalize_url(link))
        return out
