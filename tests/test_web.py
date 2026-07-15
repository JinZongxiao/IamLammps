"""Offline tests for bounded web research."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock

import httpx
import pytest

from lammps_agent.web import (
    SearchResult,
    WebResearch,
    _reject_private_destination,
    _unwrap_duckduckgo_url,
)


class _StreamResponse:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        status_code: int = 200,
        content_type: str = "text/plain",
        peer: str = "93.184.216.34",
    ) -> None:
        self._chunks = chunks
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        stream = MagicMock()
        stream.get_extra_info.return_value = (peer, 443)
        self.extensions = {"network_stream": stream}

    def __enter__(self) -> _StreamResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 300:
            raise httpx.HTTPStatusError(
                "request failed",
                request=httpx.Request("GET", "https://example.test"),
                response=httpx.Response(self.status_code),
            )

    def iter_bytes(self):
        yield from self._chunks


def _search_response(html: str) -> MagicMock:
    response = MagicMock()
    response.text = html
    stream = MagicMock()
    stream.get_extra_info.return_value = ("93.184.216.34", 443)
    response.extensions = {"network_stream": stream}
    return response


def test_search_parses_https_results_and_issues_opaque_ids(monkeypatch) -> None:
    html = """
    <a class="result__a"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.lammps.org%2Fpair_eam.html">
      EAM <b>documentation</b>
    </a>
    <div class="result__snippet">Official <b>LAMMPS</b> reference.</div>
    <a class="result__a" href="http://unsafe.example/potential">HTTP result</a>
    <div class="result__snippet">Must be ignored.</div>
    <a class="result__a" href="https://example.org/paper">Research paper</a>
    <a class="result__snippet">A useful result.</a>
    """
    post = MagicMock(return_value=_search_response(html))
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )

    research = WebResearch(timeout=3.5)
    results = research.search(" tungsten EAM ", limit=10)

    assert results == [
        SearchResult(
            "web-1",
            "EAM documentation",
            "https://docs.lammps.org/pair_eam.html",
            "Official LAMMPS reference.",
        ),
        SearchResult(
            "web-2",
            "Research paper",
            "https://example.org/paper",
            "A useful result.",
        ),
    ]
    assert WebResearch.as_dicts(results)[0]["result_id"] == "web-1"
    post.assert_called_once_with(
        "https://html.duckduckgo.com/html/",
        data={"q": "tungsten EAM"},
        headers={"User-Agent": "IamLAMMPS/5 research client"},
        timeout=3.5,
        follow_redirects=False,
    )
    post.return_value.raise_for_status.assert_called_once_with()


def test_search_rejects_empty_query_without_network(monkeypatch) -> None:
    post = MagicMock()
    monkeypatch.setattr(httpx, "post", post)

    with pytest.raises(ValueError, match="must not be empty"):
        WebResearch().search("  ")

    post.assert_not_called()


def test_search_rejects_unverifiable_connected_peer(monkeypatch) -> None:
    response = _search_response("")
    response.extensions = {}
    monkeypatch.setattr(httpx, "post", MagicMock(return_value=response))
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )

    with pytest.raises(ValueError, match="verifiable peer"):
        WebResearch().search("LAMMPS")


def test_fetch_requires_a_previously_returned_result_id(monkeypatch) -> None:
    stream = MagicMock()
    getaddrinfo = MagicMock()
    monkeypatch.setattr(httpx, "stream", stream)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)

    with pytest.raises(ValueError, match="Unknown search result ID"):
        WebResearch().fetch("https://example.org/not-an-id")

    stream.assert_not_called()
    getaddrinfo.assert_not_called()


def test_fetch_checks_destination_then_extracts_visible_html(monkeypatch) -> None:
    research = WebResearch(timeout=4, max_bytes=1_000)
    research._results["web-1"] = SearchResult(
        "web-1", "LAMMPS docs", "https://docs.lammps.org/page.html", "snippet"
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    stream = MagicMock(
        return_value=_StreamResponse(
            [
                b"<html><style>hidden</style><body>Visible ",
                b"text<script>bad()</script></body></html>",
            ],
            content_type="text/html; charset=utf-8",
        )
    )
    monkeypatch.setattr(httpx, "stream", stream)

    page = research.fetch("web-1")

    assert page == {
        "title": "LAMMPS docs",
        "url": "https://docs.lammps.org/page.html",
        "content": "Visible text",
    }
    stream.assert_called_once_with(
        "GET",
        "https://docs.lammps.org/page.html",
        headers={"User-Agent": "IamLAMMPS/5 research client"},
        timeout=4,
        follow_redirects=False,
    )


def test_fetch_rejects_private_destination_before_http(monkeypatch) -> None:
    research = WebResearch()
    research._results["web-1"] = SearchResult(
        "web-1", "internal", "https://metadata.example/", ""
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    )
    stream = MagicMock()
    monkeypatch.setattr(httpx, "stream", stream)

    with pytest.raises(ValueError, match="destinations are blocked"):
        research.fetch("web-1")

    stream.assert_not_called()


def test_fetch_rejects_peer_not_in_validated_dns_set(monkeypatch) -> None:
    research = WebResearch()
    research._results["web-1"] = SearchResult(
        "web-1", "swapped", "https://example.org/page", ""
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    monkeypatch.setattr(
        httpx,
        "stream",
        MagicMock(return_value=_StreamResponse([], peer="93.184.216.35")),
    )

    with pytest.raises(ValueError, match="does not match"):
        research.fetch("web-1")


def test_fetch_rejects_oversized_response(monkeypatch) -> None:
    research = WebResearch(max_bytes=5)
    research._results["web-1"] = SearchResult(
        "web-1", "large", "https://example.org/large", ""
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    monkeypatch.setattr(
        httpx, "stream", MagicMock(return_value=_StreamResponse([b"123", b"456"]))
    )

    with pytest.raises(ValueError, match="size limit"):
        research.fetch("web-1")


def test_fetch_does_not_follow_redirects(monkeypatch) -> None:
    research = WebResearch()
    research._results["web-1"] = SearchResult(
        "web-1", "redirect", "https://example.org/redirect", ""
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    monkeypatch.setattr(
        httpx,
        "stream",
        MagicMock(return_value=_StreamResponse([], status_code=302)),
    )

    with pytest.raises(httpx.HTTPStatusError):
        research.fetch("web-1")


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org/",
        "file:///etc/passwd",
        "https:///missing-host",
    ],
)
def test_ssrf_guard_requires_https_with_a_hostname(url: str) -> None:
    with pytest.raises(ValueError, match="Only public HTTPS"):
        _reject_private_destination(url)


def test_ssrf_guard_rejects_if_any_resolved_address_is_not_global(monkeypatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 443, 0, 0)),
        ],
    )

    with pytest.raises(ValueError, match="destinations are blocked"):
        _reject_private_destination("https://example.org")


def test_duckduckgo_unwrap_leaves_direct_urls_unchanged() -> None:
    assert _unwrap_duckduckgo_url("https://example.org/page?q=1") == (
        "https://example.org/page?q=1"
    )
