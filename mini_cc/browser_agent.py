from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from pydantic import BaseModel


class BrowserResult(BaseModel):
    url: str
    title: str = ""
    text: str = ""
    backend: str = "urllib"


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ""
        self._in_title = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        if tag.lower() == "title":
            self._in_title = True

    def handle_endtag(self, tag: str):
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str):
        stripped = " ".join(data.split())
        if not stripped:
            return
        if self._in_title:
            self.title += stripped
        else:
            self.parts.append(stripped)


class BrowserAgent:
    """Browser/document reader adapter.

    Production deployments can replace this with Playwright, Browser Use, or
    Crawl4AI. The default backend is dependency-free and read-only.
    """

    def fetch(self, url: str, timeout: int = 20) -> BrowserResult:
        request = Request(url, headers={"User-Agent": "mini-claude-code-v3"})
        with urlopen(request, timeout=timeout) as response:
            html = response.read().decode("utf-8", errors="ignore")
        parser = _TextExtractor()
        parser.feed(html)
        return BrowserResult(url=url, title=parser.title, text="\n".join(parser.parts[:300]))

    def search_url(self, query: str) -> str:
        return f"https://www.google.com/search?q={quote_plus(query)}"
