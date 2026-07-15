"""Offline tests for official potential discovery and installation."""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from lammps_agent.potential_catalog import (
    LammpsPotentialCatalog,
    PotentialCandidate,
)


class _StreamResponse:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        status_code: int = 200,
        peer: str = "93.184.216.34",
    ) -> None:
        self._chunks = chunks
        self.status_code = status_code
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
                request=httpx.Request("GET", "https://raw.githubusercontent.com"),
                response=httpx.Response(self.status_code),
            )

    def iter_bytes(self):
        yield from self._chunks


def _candidate(
    *,
    candidate_id: str = "lammps-1",
    filename: str = "W.eam.alloy",
    url: str = "https://raw.githubusercontent.com/lammps/lammps/stable/potentials/W.eam.alloy",
    size: int = 4,
    git_blob_sha: str | None = None,
) -> PotentialCandidate:
    if git_blob_sha is None:
        git_blob_sha = _git_blob_sha(b"data")
    return PotentialCandidate(
        candidate_id=candidate_id,
        filename=filename,
        source="LAMMPS official GitHub release",
        revision="stable_22Jul2025_update3",
        download_url=url,
        git_blob_sha=git_blob_sha,
        size=size,
    )


def _git_blob_sha(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )


def test_search_uses_latest_official_release_and_filters_listing(monkeypatch) -> None:
    release = MagicMock()
    release_stream = MagicMock()
    release_stream.get_extra_info.return_value = ("93.184.216.34", 443)
    release.extensions = {"network_stream": release_stream}
    release.json.return_value = {"tag_name": "stable_22Jul2025_update3"}
    listing = MagicMock()
    listing_stream = MagicMock()
    listing_stream.get_extra_info.return_value = ("93.184.216.34", 443)
    listing.extensions = {"network_stream": listing_stream}
    listing.json.return_value = [
        {
            "name": "AlCu.eam.alloy",
            "download_url": "https://raw.githubusercontent.com/lammps/lammps/rev/AlCu.eam.alloy",
            "sha": "abc",
            "size": 123,
        },
        {"name": "README", "download_url": None, "sha": "def", "size": 50},
        {
            "name": "Cu_u3.eam",
            "download_url": "https://raw.githubusercontent.com/lammps/lammps/rev/Cu_u3.eam",
            "sha": "ghi",
            "size": 456,
        },
        {
            "name": "Fe.eam",
            "download_url": "https://raw.githubusercontent.com/lammps/lammps/rev/Fe.eam",
            "sha": "jkl",
            "size": 789,
        },
    ]
    get = MagicMock(side_effect=[release, listing])
    monkeypatch.setattr(httpx, "get", get)

    catalog = LammpsPotentialCatalog(timeout=7)
    matches = catalog.search(" CU ", limit=1)

    assert matches == [
        PotentialCandidate(
            candidate_id="lammps-1",
            filename="AlCu.eam.alloy",
            source="LAMMPS official GitHub release",
            revision="stable_22Jul2025_update3",
            download_url="https://raw.githubusercontent.com/lammps/lammps/rev/AlCu.eam.alloy",
            git_blob_sha="abc",
            size=123,
        )
    ]
    assert get.call_args_list[1].kwargs["params"] == {
        "ref": "stable_22Jul2025_update3"
    }
    assert get.call_args_list[1].kwargs["timeout"] == 7
    release.raise_for_status.assert_called_once_with()
    listing.raise_for_status.assert_called_once_with()


def test_search_rejects_empty_query_before_network(monkeypatch) -> None:
    get = MagicMock()
    monkeypatch.setattr(httpx, "get", get)

    with pytest.raises(ValueError, match="must not be empty"):
        LammpsPotentialCatalog().search("\t")

    get.assert_not_called()


def test_install_requires_a_candidate_returned_by_search(
    tmp_path: Path, monkeypatch
) -> None:
    stream = MagicMock()
    monkeypatch.setattr(httpx, "stream", stream)

    with pytest.raises(ValueError, match="Unknown potential candidate ID"):
        LammpsPotentialCatalog().install("https://attacker.example/file", tmp_path)

    stream.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "http://raw.githubusercontent.com/lammps/lammps/file",
        "https://github.com/lammps/lammps/file",
        "https://raw.githubusercontent.com.attacker.example/file",
    ],
)
def test_install_enforces_https_download_host_allowlist(
    url: str, tmp_path: Path, monkeypatch
) -> None:
    catalog = LammpsPotentialCatalog()
    catalog._candidates["lammps-1"] = _candidate(url=url)
    stream = MagicMock()
    monkeypatch.setattr(httpx, "stream", stream)

    with pytest.raises(ValueError, match="host is not allowlisted"):
        catalog.install("lammps-1", tmp_path)

    stream.assert_not_called()


def test_install_rejects_declared_oversize_before_creating_destination(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "potentials"
    catalog = LammpsPotentialCatalog(max_bytes=3)
    catalog._candidates["lammps-1"] = _candidate(size=4)
    stream = MagicMock()
    monkeypatch.setattr(httpx, "stream", stream)

    with pytest.raises(ValueError, match="configured size limit"):
        catalog.install("lammps-1", destination)

    assert not destination.exists()
    stream.assert_not_called()


def test_install_streams_atomically_and_writes_auditable_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "potentials"
    payload = b"EAM\ndata\n"
    candidate = _candidate(
        filename="../W.eam.alloy",
        size=len(payload),
        git_blob_sha=_git_blob_sha(payload),
    )
    catalog = LammpsPotentialCatalog(max_bytes=100)
    catalog._candidates[candidate.candidate_id] = candidate
    stream = MagicMock(return_value=_StreamResponse([payload[:3], payload[3:]]))
    monkeypatch.setattr(httpx, "stream", stream)

    manifest = catalog.install(candidate.candidate_id, destination)

    target = destination / "W.eam.alloy"
    provenance = destination / "W.eam.alloy.provenance.json"
    assert target.read_bytes() == payload
    assert target.stat().st_mode & 0o777 == 0o644
    assert list(destination.iterdir()) == [target, provenance]
    assert manifest["filename"] == "../W.eam.alloy"
    assert manifest["installed_path"] == "W.eam.alloy"
    assert manifest["bytes"] == len(payload)
    assert manifest["sha256"] == hashlib.sha256(payload).hexdigest()
    assert json.loads(provenance.read_text(encoding="utf-8")) == manifest
    stream.assert_called_once_with(
        "GET",
        candidate.download_url,
        headers={"User-Agent": "IamLAMMPS/5"},
        timeout=20,
        follow_redirects=False,
    )


def test_install_rejects_payload_not_matching_candidate_git_blob_sha(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "potentials"
    payload = b"data"
    catalog = LammpsPotentialCatalog(max_bytes=100)
    catalog._candidates["lammps-1"] = _candidate(
        size=len(payload),
        git_blob_sha=_git_blob_sha(b"evil"),
    )
    monkeypatch.setattr(
        httpx,
        "stream",
        MagicMock(return_value=_StreamResponse([payload])),
    )

    with pytest.raises(ValueError, match="Git blob SHA"):
        catalog.install("lammps-1", destination)

    assert list(destination.iterdir()) == []


def test_install_rejects_payload_size_not_matching_candidate_declaration(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "potentials"
    payload = b"data"
    catalog = LammpsPotentialCatalog(max_bytes=100)
    catalog._candidates["lammps-1"] = _candidate(
        size=len(payload) + 1,
        git_blob_sha=_git_blob_sha(payload),
    )
    monkeypatch.setattr(
        httpx,
        "stream",
        MagicMock(return_value=_StreamResponse([payload])),
    )

    with pytest.raises(ValueError, match="declared size"):
        catalog.install("lammps-1", destination)

    assert list(destination.iterdir()) == []


def test_install_cleans_temporary_file_when_actual_payload_exceeds_limit(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "potentials"
    catalog = LammpsPotentialCatalog(max_bytes=5)
    catalog._candidates["lammps-1"] = _candidate(size=1)
    monkeypatch.setattr(
        httpx,
        "stream",
        MagicMock(return_value=_StreamResponse([b"123", b"456"])),
    )

    with pytest.raises(ValueError, match="exceeds the size limit"):
        catalog.install("lammps-1", destination)

    assert list(destination.iterdir()) == []


def test_install_rejects_redirect_and_cleans_temporary_file(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "potentials"
    catalog = LammpsPotentialCatalog()
    catalog._candidates["lammps-1"] = _candidate()
    monkeypatch.setattr(
        httpx,
        "stream",
        MagicMock(return_value=_StreamResponse([], status_code=302)),
    )

    with pytest.raises(httpx.HTTPStatusError):
        catalog.install("lammps-1", destination)

    assert list(destination.iterdir()) == []


def test_install_refuses_to_overwrite_existing_potential(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "W.eam.alloy"
    target.write_text("existing", encoding="utf-8")
    catalog = LammpsPotentialCatalog()
    catalog._candidates["lammps-1"] = _candidate()
    stream = MagicMock()
    monkeypatch.setattr(httpx, "stream", stream)

    with pytest.raises(FileExistsError, match="Potential already exists"):
        catalog.install("lammps-1", tmp_path)

    assert target.read_text(encoding="utf-8") == "existing"
    stream.assert_not_called()
