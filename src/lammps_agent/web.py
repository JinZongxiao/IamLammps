"""Bounded web research used by the simulation agent."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx


@dataclass(frozen=True)
class SearchResult:
    result_id: str
    title: str
    url: str
    snippet: str


class _SearchParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[tuple[str, str, str]] = []
        self._url = ""
        self._title: list[str] = []
        self._snippet: list[str] = []
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            self._url = _unwrap_duckduckgo_url(values.get("href") or "")
            self._title = []
            self._in_title = True
        elif tag in {"a", "div"} and "result__snippet" in classes:
            self._snippet = []
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
        if tag in {"a", "div"} and self._in_snippet:
            self._in_snippet = False
            title = " ".join("".join(self._title).split())
            snippet = " ".join("".join(self._snippet).split())
            if self._url and title:
                self.results.append((title, self._url, snippet))
            self._url = ""

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title.append(data)
        if self._in_snippet:
            self._snippet.append(data)


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignored += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored:
            self._ignored -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored and data.strip():
            self.parts.append(data.strip())


class WebResearch:
    """Search the public web and fetch only previously returned HTTPS results."""

    def __init__(self, timeout: float = 15.0, max_bytes: int = 1_000_000) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self._results: dict[str, SearchResult] = {}

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("Search query must not be empty")
        limit = max(1, min(limit, 10))
        endpoint = "https://html.duckduckgo.com/html/"
        approved_addresses = _public_addresses(endpoint)
        response = httpx.post(
            endpoint,
            data={"q": query},
            headers={"User-Agent": "IamLAMMPS/5 research client"},
            timeout=self.timeout,
            follow_redirects=False,
        )
        _verify_response_peer(response, approved_addresses)
        response.raise_for_status()
        parser = _SearchParser()
        parser.feed(response.text)
        results: list[SearchResult] = []
        for title, url, snippet in parser.results:
            if not url.startswith("https://"):
                continue
            result_id = f"web-{len(self._results) + 1}"
            result = SearchResult(result_id, title, url, snippet)
            self._results[result_id] = result
            results.append(result)
            if len(results) >= limit:
                break
        return results

    def fetch(
        self,
        result_id: str,
        max_chars: int = 12_000,
        expected_url: str | None = None,
    ) -> dict[str, str]:
        result = self._results.get(result_id)
        if result is None:
            raise ValueError("Unknown search result ID; call web_search first")
        if expected_url is not None and result.url != expected_url:
            raise ValueError("Search result URL does not match the approved URL")
        approved_addresses = _public_addresses(result.url)
        max_chars = max(500, min(max_chars, 30_000))
        with httpx.stream(
            "GET",
            result.url,
            headers={"User-Agent": "IamLAMMPS/5 research client"},
            timeout=self.timeout,
            follow_redirects=False,
        ) as response:
            _verify_response_peer(response, approved_addresses)
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_bytes():
                content.extend(chunk)
                if len(content) > self.max_bytes:
                    raise ValueError("Web page exceeds the configured size limit")
            media_type = response.headers.get("content-type", "").lower()
        text = bytes(content).decode("utf-8", errors="replace")
        if "html" in media_type:
            parser = _TextParser()
            parser.feed(text)
            text = "\n".join(parser.parts)
        return {
            "title": result.title,
            "url": result.url,
            "content": text[:max_chars],
        }

    @staticmethod
    def as_dicts(results: list[SearchResult]) -> list[dict[str, str]]:
        return [asdict(result) for result in results]


def _unwrap_duckduckgo_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return unquote(target[0])
    return url


def _reject_private_destination(url: str) -> None:
    _public_addresses(url)


def _public_addresses(url: str) -> frozenset[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Only public HTTPS pages may be fetched")
    addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for info in socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM):
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise ValueError("Private, loopback, and link-local destinations are blocked")
        addresses.add(address)
    if not addresses:
        raise ValueError("Destination did not resolve to a public address")
    return frozenset(addresses)


def _verify_response_peer(response: Any, approved_addresses: frozenset) -> None:
    extensions = getattr(response, "extensions", None)
    stream = extensions.get("network_stream") if isinstance(extensions, dict) else None
    peer = stream.get_extra_info("server_addr") if stream is not None else None
    if not isinstance(peer, tuple) or not peer:
        raise ValueError("HTTP connection did not expose a verifiable peer address")
    try:
        address = ipaddress.ip_address(peer[0])
    except ValueError as exc:
        raise ValueError("HTTP connection exposed an invalid peer address") from exc
    if not address.is_global or address not in approved_addresses:
        raise ValueError("HTTP peer does not match the validated public destination")
