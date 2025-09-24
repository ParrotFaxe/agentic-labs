"""TradingEdge MCP server.

This MCP server exposes a tool that authenticates against https://tradingedge.club
by reusing an authenticated browser session cookie.  After authentication, the
server crawls the site's articles, extracts ticker references, and aggregates the
key insights into a Markdown table so downstream agents can reason about them.

The server intentionally avoids performing credential-based logins because the
site relies on single sign-on.  Operators must provide a valid session cookie via
``TRADINGEDGE_SESSION_COOKIE`` (for example by copying it from their browser's
storage).  Optional environment variables allow overriding the base URL,
article index URL, or a JSON API endpoint if the deployment exposes one.

Because the lab environment might not have outbound network access, the tool
surfaces informative error messages so an operator understands why collection
failed.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from http.cookies import SimpleCookie
from typing import Dict, Iterable, List, Sequence
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

DEFAULT_BASE_URL = "https://tradingedge.club"


@dataclass
class Article:
    """Normalized article data returned from the TradingEdge site."""

    title: str
    url: str
    published_at: str
    content: str


class TradingEdgeClient:
    """Client responsible for collecting article data using a session cookie."""

    def __init__(
        self,
        session_cookie: str,
        api_endpoint: str | None = None,
        index_url: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        if not session_cookie:
            raise ValueError("A TradingEdge session cookie is required for authentication.")

        self.session_cookie = session_cookie
        self.api_endpoint = api_endpoint
        self.session = requests.Session()

        parsed_base = urlparse(base_url)
        if not parsed_base.scheme or not parsed_base.netloc:
            base_url = DEFAULT_BASE_URL
            parsed_base = urlparse(base_url)
        self.base_url = f"{parsed_base.scheme}://{parsed_base.netloc}"

        # Derive the canonical base URL from whichever endpoint is provided.
        for candidate in (api_endpoint, index_url, base_url):
            if not candidate:
                continue
            parsed = urlparse(candidate)
            if parsed.scheme and parsed.netloc:
                self.base_url = f"{parsed.scheme}://{parsed.netloc}"
                break

        self.index_url = index_url or f"{self.base_url}/articles"

        self._apply_session_cookie()

    def _apply_session_cookie(self) -> None:
        """Load a user-supplied cookie string into the current session."""

        cookie_jar = SimpleCookie()
        cookie_jar.load(self.session_cookie)
        domain = urlparse(self.base_url).netloc or "tradingedge.club"
        for name, morsel in cookie_jar.items():
            self.session.cookies.set(name, morsel.value, domain=domain)

    def _normalize_url(self, url: str) -> str:
        if not url:
            return self.base_url
        return urljoin(self.base_url + "/", url)

    def _download_article(
        self,
        url: str,
        *,
        title_hint: str = "",
        published_hint: str = "",
    ) -> Article:
        """Download and normalise a single article page."""

        resolved_url = self._normalize_url(url)
        response = self.session.get(resolved_url, timeout=30)
        if response.status_code in (401, 403):
            raise RuntimeError("TradingEdge session cookie was rejected while fetching article content.")
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        article_container = soup.find("article")
        if article_container:
            body_text = article_container.get_text(" ", strip=True)
        else:
            main_container = soup.find("main")
            if main_container:
                body_text = main_container.get_text(" ", strip=True)
            else:
                body_text = soup.get_text(" ", strip=True)

        title = ""
        title_tag = soup.find("h1")
        if title_tag:
            title = title_tag.get_text(strip=True)
        if not title and soup.title:
            title = soup.title.get_text(strip=True)
        if not title:
            title = title_hint or resolved_url

        published_at = ""
        time_tag = soup.find("time")
        if time_tag:
            published_at = time_tag.get("datetime") or time_tag.get_text(strip=True)
        if not published_at:
            published_at = published_hint

        return Article(title=title, url=resolved_url, published_at=published_at or "", content=body_text)

    def _fetch_articles_via_api(self, limit: int | None = None) -> List[Article]:
        """Attempt to retrieve articles through a JSON API endpoint."""

        if not self.api_endpoint:
            raise RuntimeError("No TradingEdge API endpoint configured.")

        params: Dict[str, int] = {}
        if limit is not None:
            params["limit"] = limit
            params.setdefault("per_page", limit)

        response = self.session.get(self.api_endpoint, params=params or None, timeout=30)
        if response.status_code in (401, 403):
            raise RuntimeError("TradingEdge session cookie was rejected by the API endpoint.")
        response.raise_for_status()

        try:
            payload = response.json()
        except ValueError as exc:  # pragma: no cover - defensive
            raise RuntimeError("TradingEdge API returned invalid JSON data.") from exc

        if isinstance(payload, dict):
            if isinstance(payload.get("articles"), list):
                records = payload["articles"]
            elif isinstance(payload.get("data"), list):
                records = payload["data"]
            elif isinstance(payload.get("items"), list):
                records = payload["items"]
            else:
                raise RuntimeError("TradingEdge API response did not include an article collection.")
        elif isinstance(payload, list):
            records = payload
        else:
            raise RuntimeError("TradingEdge API response format is not supported.")

        articles: List[Article] = []
        for record in records:
            if limit is not None and len(articles) >= limit:
                break
            if not isinstance(record, dict):
                continue

            title = str(
                record.get("title")
                or record.get("headline")
                or record.get("name")
                or "Untitled"
            ).strip()
            url = record.get("url") or record.get("link") or record.get("permalink")
            if not url:
                slug = record.get("slug")
                if isinstance(slug, str) and slug.strip():
                    url = slug
            resolved_url = self._normalize_url(url) if url else ""

            published_at = (
                record.get("published_at")
                or record.get("date")
                or record.get("created_at")
                or record.get("updated_at")
                or ""
            )

            content = record.get("content") or record.get("body") or record.get("summary") or ""
            text_content = ""
            if isinstance(content, str) and content.strip():
                text_content = BeautifulSoup(content, "html.parser").get_text(" ", strip=True)

            if text_content:
                articles.append(
                    Article(
                        title=title,
                        url=resolved_url or self.base_url,
                        published_at=str(published_at or ""),
                        content=text_content,
                    )
                )
            elif resolved_url:
                articles.append(
                    self._download_article(
                        resolved_url,
                        title_hint=title,
                        published_hint=str(published_at or ""),
                    )
                )
        return articles

    def _extract_article_links(self, soup: BeautifulSoup, page_url: str) -> List[str]:
        """Identify likely article links from an index page."""

        links: List[str] = []
        for article_tag in soup.find_all("article"):
            anchor = article_tag.find("a", href=True)
            if anchor:
                links.append(urljoin(page_url, anchor["href"]))

        if not links:
            for anchor in soup.find_all("a", href=True):
                href = anchor["href"]
                if href.startswith("#"):
                    continue
                absolute = urljoin(page_url, href)
                if urlparse(absolute).netloc != urlparse(self.base_url).netloc:
                    continue
                if any(keyword in absolute.lower() for keyword in ("article", "analysis", "blog", "idea", "report", "update")):
                    links.append(absolute)
        return links

    def _find_next_page(self, soup: BeautifulSoup, current_url: str) -> str | None:
        """Locate the next pagination link, if available."""

        next_anchor = soup.find("a", attrs={"rel": ["next"]})
        if not next_anchor:
            for candidate in soup.find_all("a", href=True):
                text = candidate.get_text(strip=True).lower()
                if text in {"next", "older posts", "older", "more", "load more"}:
                    next_anchor = candidate
                    break
        if next_anchor and next_anchor.get("href"):
            return urljoin(current_url, next_anchor["href"])
        return None

    def _fetch_articles_via_scraping(self, limit: int | None = None) -> List[Article]:
        """Fallback that scrapes article pages directly."""

        if not self.index_url:
            raise RuntimeError("No TradingEdge index URL configured for scraping.")

        articles: List[Article] = []
        visited: set[str] = set()
        next_url: str | None = self._normalize_url(self.index_url)

        while next_url and (limit is None or len(articles) < limit):
            response = self.session.get(next_url, timeout=30)
            if response.status_code in (401, 403):
                raise RuntimeError("TradingEdge session cookie was rejected while accessing article index pages.")
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            candidate_links = self._extract_article_links(soup, next_url)

            for link in candidate_links:
                if limit is not None and len(articles) >= limit:
                    break
                normalized = self._normalize_url(link)
                if normalized in visited:
                    continue
                visited.add(normalized)
                try:
                    article = self._download_article(normalized)
                except Exception:
                    continue
                articles.append(article)

            next_url = self._find_next_page(soup, next_url)

        return articles

    def fetch_articles(self, limit: int | None = None) -> List[Article]:
        """Fetch TradingEdge articles using the available strategy."""

        errors: List[str] = []
        articles: List[Article] = []

        if self.api_endpoint:
            try:
                articles = self._fetch_articles_via_api(limit=limit)
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(str(exc))

        if not articles:
            try:
                articles = self._fetch_articles_via_scraping(limit=limit)
            except Exception as exc:
                errors.append(str(exc))

        if not articles:
            detail = "; ".join(errors) if errors else "no articles were discovered"
            raise RuntimeError(f"Unable to fetch TradingEdge articles: {detail}.")

        return articles


def extract_tickers(text: str) -> Sequence[str]:
    """Extract likely ticker symbols from the provided article text."""

    tickers = set()

    # $TICKER format
    for match in re.findall(r"\$([A-Z]{1,5})", text):
        tickers.add(match)

    # Plain uppercase ticker mentions (1-5 letters) surrounded by word boundaries.
    for match in re.findall(r"\b([A-Z]{1,5})\b", text):
        if match not in COMMON_UPPERCASE_STOP_WORDS:
            tickers.add(match)

    return sorted(tickers)


COMMON_UPPERCASE_STOP_WORDS = {
    "CEO",
    "CFO",
    "EPS",
    "ETF",
    "GDP",
    "AI",
    "IPO",
    "USD",
    "FED",
    "FOMC",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "YOY",
    "PCE",
    "PMI",
    "EV",
    "ETF",
}


def aggregate_article_insights(articles: Iterable[Article]) -> Dict[str, List[Dict[str, str]]]:
    """Map tickers to the key context captured inside each article."""

    ticker_to_insights: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for article in articles:
        tickers = extract_tickers(article.content)
        if not tickers:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", article.content)
        for ticker in tickers:
            relevant_sentences = [s for s in sentences if ticker in s]
            if not relevant_sentences:
                continue
            snippet = " ".join(relevant_sentences)[:400].strip()
            ticker_to_insights[ticker].append(
                {
                    "article": article.title,
                    "url": article.url,
                    "published_at": article.published_at,
                    "insight": snippet,
                }
            )

    return dict(ticker_to_insights)


def build_markdown_table(ticker_insights: Dict[str, List[Dict[str, str]]]) -> str:
    """Create a Markdown table summarising ticker insights."""

    if not ticker_insights:
        return "| Ticker | Key Takeaways |\n| --- | --- |\n| - | No ticker mentions were detected. |"

    rows = ["| Ticker | Key Takeaways |", "| --- | --- |"]
    for ticker in sorted(ticker_insights):
        insights = ticker_insights[ticker]
        bullet_lines = []
        for insight in insights:
            bullet_lines.append(
                f"- **{insight['article']}** ({insight['published_at']}): {insight['insight']} "
                f"[[link]]({insight['url']})"
            )
        rows.append(f"| {ticker} | {'<br>'.join(bullet_lines)} |")
    return "\n".join(rows)


mcp = FastMCP("TradingEdge Intelligence", host="0.0.0.0", port=8010)


@mcp.tool()
def tradingedge_ticker_digest(limit: int | None = None) -> Dict[str, str]:
    """Collect TradingEdge articles and summarise ticker information.

    Args:
        limit: Optional maximum number of recent posts to ingest.

    Returns:
        A dictionary with two keys: ``status`` describing the operation outcome
        and ``table`` containing a Markdown formatted table with the ticker
        insights.
    """

    session_cookie = os.getenv("TRADINGEDGE_SESSION_COOKIE")
    if not session_cookie:
        raise RuntimeError("Provide TRADINGEDGE_SESSION_COOKIE with a valid TradingEdge session cookie.")

    client = TradingEdgeClient(
        session_cookie=session_cookie,
        api_endpoint=os.getenv("TRADINGEDGE_API_ENDPOINT"),
        index_url=os.getenv("TRADINGEDGE_INDEX_URL"),
        base_url=os.getenv("TRADINGEDGE_BASE_URL", DEFAULT_BASE_URL),
    )
    articles = client.fetch_articles(limit=limit)
    ticker_insights = aggregate_article_insights(articles)
    table = build_markdown_table(ticker_insights)
    return {"status": "success", "table": table, "tickers_found": sorted(ticker_insights.keys())}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
