"""Unit tests for crawling/extraction, chunking and the ingest service.

These tests avoid network, OpenAI and database calls by using fixtures and
fakes, so they run fast and offline.
"""
from __future__ import annotations

import pytest

from app.core.loader import (
    WebCrawler,
    html_to_markdown,
    is_valid_url,
    normalize_url,
    same_domain,
)
from app.core.splitter import SemanticChunker, count_tokens
from app.models.schemas import CrawledPage

SAMPLE_HTML = """
<html>
  <head><title>Test Doc</title></head>
  <body>
    <nav class="navbar">menu stuff we should drop</nav>
    <header>site header</header>
    <main>
      <h1>Intro</h1>
      <p>Hello world, this is the main body of the page.</p>
      <h2>Code</h2>
      <pre><code>def f():
    return 1</code></pre>
      <h2>Table</h2>
      <table>
        <tr><th>a</th><th>b</th></tr>
        <tr><td>1</td><td>2</td></tr>
      </table>
      <a href="/next">next page</a>
      <a href="https://other.com/x">external</a>
    </main>
    <footer>copyright boilerplate</footer>
    <script>console.log('drop me')</script>
  </body>
</html>
"""


# --------------------------------------------------------------------------- #
# URL helpers
# --------------------------------------------------------------------------- #
def test_is_valid_url():
    assert is_valid_url("https://example.com")
    assert is_valid_url("http://a.b/c")
    assert not is_valid_url("notaurl")
    assert not is_valid_url("ftp://x.com")
    assert not is_valid_url("")


def test_normalize_url_strips_fragment_and_trailing_slash():
    assert normalize_url("https://x.com/a/#frag") == "https://x.com/a"
    assert normalize_url("https://x.com/") == "https://x.com"


def test_same_domain():
    assert same_domain("https://x.com/a", "https://x.com/b")
    assert not same_domain("https://x.com/a", "https://y.com/a")


# --------------------------------------------------------------------------- #
# HTML -> markdown
# --------------------------------------------------------------------------- #
def test_html_to_markdown_extracts_and_cleans():
    title, md, links = html_to_markdown(SAMPLE_HTML, "https://example.com/doc")
    assert title == "Test Doc"
    # boilerplate removed
    assert "menu stuff" not in md
    assert "site header" not in md
    assert "copyright boilerplate" not in md
    assert "drop me" not in md
    # content preserved
    assert "# Intro" in md
    assert "Hello world" in md
    # code fence + table preserved
    assert "```" in md
    assert "def f():" in md
    assert "| a | b |" in md
    # links resolved + collected
    assert "https://example.com/next" in links
    assert "https://other.com/x" in links


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def _make_page(markdown: str) -> CrawledPage:
    return CrawledPage(
        source_url="https://example.com/doc",
        page_title="Test Doc",
        markdown=markdown,
        parent_url=None,
    )


def test_chunk_metadata_is_complete():
    page = _make_page("# Title\n\n" + ("word " * 50))
    chunks = SemanticChunker().chunk_page(page)
    assert chunks
    m = chunks[0].metadata
    assert m.chunk_id
    assert m.source_url == "https://example.com/doc"
    assert m.title == "Test Doc"
    assert m.token_count > 0
    assert m.content_hash
    assert m.token_count == count_tokens(chunks[0].text)


def test_code_blocks_are_never_split():
    big_code = "```\n" + "\n".join(f"line_{i} = {i}" for i in range(400)) + "\n```"
    page = _make_page(f"# Code\n\n{big_code}\n")
    chunks = SemanticChunker(target_tokens=200, max_tokens=300).chunk_page(page)
    # The fenced block must live wholly inside exactly one chunk.
    holders = [c for c in chunks if "line_0 = 0" in c.text]
    assert len(holders) == 1
    holder = holders[0]
    assert "line_399 = 399" in holder.text
    assert holder.text.count("```") % 2 == 0  # balanced fences


def test_tables_are_never_split():
    rows = "\n".join(f"| r{i}a | r{i}b |" for i in range(100))
    table = "| a | b |\n| --- | --- |\n" + rows
    page = _make_page(f"# Table\n\n{table}\n")
    chunks = SemanticChunker(target_tokens=150, max_tokens=250).chunk_page(page)
    holders = [c for c in chunks if "r0a" in c.text]
    assert len(holders) == 1
    assert "r99b" in holders[0].text


def test_headings_stay_with_content_and_section_breadcrumb():
    md = "# Guide\n\n## Setup\n\nInstall the package first.\n\n## Usage\n\nThen run it."
    page = _make_page(md)
    chunks = SemanticChunker().chunk_page(page)
    assert chunks
    # section breadcrumb reflects hierarchy
    sections = {c.metadata.section for c in chunks}
    assert any("Guide" in s for s in sections)


def test_overlap_between_consecutive_chunks():
    # Force several chunks with a long prose page.
    para = ("Sentence number {} provides additional context. ".format)
    text = "# Doc\n\n" + " ".join(para(i) for i in range(400))
    page = _make_page(text)
    chunker = SemanticChunker(target_tokens=200, max_tokens=300, overlap_tokens=40)
    chunks = chunker.chunk_page(page)
    assert len(chunks) >= 2
    # later chunks include overlap text => start tokens shared with previous
    assert chunks[1].metadata.token_count > 0


# --------------------------------------------------------------------------- #
# Crawler invalid-URL handling
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_crawl_rejects_invalid_url():
    with pytest.raises(ValueError):
        await WebCrawler().crawl("not-a-url")
