"""Auditable discovery and installation of official LAMMPS potential files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from .web import _public_addresses, _verify_response_peer

_API_ROOT = "https://api.github.com/repos/lammps/lammps"
_ALLOWED_DOWNLOAD_HOSTS = {"raw.githubusercontent.com"}
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class PotentialCandidate:
    candidate_id: str
    filename: str
    source: str
    revision: str
    download_url: str
    git_blob_sha: str
    size: int


class LammpsPotentialCatalog:
    """Search a pinned official LAMMPS release and install selected data files."""

    def __init__(self, timeout: float = 20.0, max_bytes: int = 50_000_000) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self._candidates: dict[str, PotentialCandidate] = {}

    def search(self, query: str, limit: int = 10) -> list[PotentialCandidate]:
        query = query.strip().casefold()
        if not query:
            raise ValueError("Potential query must not be empty")
        release_url = f"{_API_ROOT}/releases/latest"
        release_addresses = _public_addresses(release_url)
        release = httpx.get(
            release_url,
            headers={"User-Agent": "IamLAMMPS/5"},
            timeout=self.timeout,
        )
        _verify_response_peer(release, release_addresses)
        release.raise_for_status()
        revision = release.json()["tag_name"]
        listing_url = f"{_API_ROOT}/contents/potentials"
        listing_addresses = _public_addresses(listing_url)
        listing = httpx.get(
            listing_url,
            params={"ref": revision},
            headers={"User-Agent": "IamLAMMPS/5"},
            timeout=self.timeout,
        )
        _verify_response_peer(listing, listing_addresses)
        listing.raise_for_status()
        matches: list[PotentialCandidate] = []
        for item in listing.json():
            filename = str(item.get("name", ""))
            download_url = item.get("download_url")
            if query not in filename.casefold() or not download_url:
                continue
            candidate_id = f"lammps-{len(self._candidates) + 1}"
            candidate = PotentialCandidate(
                candidate_id=candidate_id,
                filename=filename,
                source="LAMMPS official GitHub release",
                revision=revision,
                download_url=str(download_url),
                git_blob_sha=str(item.get("sha", "")),
                size=int(item.get("size", 0)),
            )
            self._candidates[candidate_id] = candidate
            matches.append(candidate)
            if len(matches) >= max(1, min(limit, 20)):
                break
        return matches

    def install(self, candidate_id: str, destination: Path) -> dict[str, object]:
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            raise ValueError("Unknown potential candidate ID; call search_potentials first")
        parsed = httpx.URL(candidate.download_url)
        if parsed.scheme != "https" or parsed.host not in _ALLOWED_DOWNLOAD_HOSTS:
            raise ValueError("Potential artifact host is not allowlisted")
        if candidate.size > self.max_bytes:
            raise ValueError("Potential artifact exceeds the configured size limit")
        if candidate.size < 0:
            raise ValueError("Potential artifact has an invalid declared size")
        if _GIT_SHA_RE.fullmatch(candidate.git_blob_sha) is None:
            raise ValueError("Potential candidate has an invalid Git blob SHA")
        approved_addresses = _public_addresses(candidate.download_url)
        if destination.exists() and destination.is_symlink():
            raise ValueError("Potential destination must not be a symlink")
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / Path(candidate.filename).name
        if target.exists():
            raise FileExistsError(f"Potential already exists: {target.name}")

        digest = hashlib.sha256()
        git_blob_digest = hashlib.sha1(usedforsecurity=False)
        git_blob_digest.update(f"blob {candidate.size}\0".encode("ascii"))
        total = 0
        with tempfile.NamedTemporaryFile(dir=destination, delete=False) as temporary:
            temp_path = Path(temporary.name)
            try:
                with httpx.stream(
                    "GET",
                    candidate.download_url,
                    headers={"User-Agent": "IamLAMMPS/5"},
                    timeout=self.timeout,
                    follow_redirects=False,
                ) as response:
                    _verify_response_peer(response, approved_addresses)
                    response.raise_for_status()
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > self.max_bytes:
                            raise ValueError("Potential artifact exceeds the size limit")
                        digest.update(chunk)
                        git_blob_digest.update(chunk)
                        temporary.write(chunk)
                temporary.flush()
                if total != candidate.size:
                    raise ValueError(
                        "Potential artifact does not match its declared size"
                    )
                if git_blob_digest.hexdigest() != candidate.git_blob_sha:
                    raise ValueError(
                        "Potential artifact does not match its Git blob SHA"
                    )
                target_hash = digest.hexdigest()
            except Exception:
                temp_path.unlink(missing_ok=True)
                raise
        manifest = {
            **asdict(candidate),
            "sha256": target_hash,
            "installed_path": target.name,
            "bytes": total,
        }
        manifest_path = destination / f"{target.name}.provenance.json"
        if manifest_path.exists() or manifest_path.is_symlink():
            temp_path.unlink(missing_ok=True)
            raise FileExistsError(f"Potential provenance already exists: {manifest_path.name}")
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination, delete=False
        ) as manifest_temporary:
            manifest_temp = Path(manifest_temporary.name)
            json.dump(manifest, manifest_temporary, ensure_ascii=False, indent=2)
            manifest_temporary.write("\n")
            manifest_temporary.flush()
            os.fsync(manifest_temporary.fileno())
        try:
            os.link(temp_path, target, follow_symlinks=False)
            temp_path.unlink()
            target.chmod(0o644)
            os.link(manifest_temp, manifest_path, follow_symlinks=False)
            manifest_temp.unlink()
        except Exception:
            temp_path.unlink(missing_ok=True)
            manifest_temp.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            raise
        return manifest

    def get_candidate(self, candidate_id: str) -> PotentialCandidate:
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            raise ValueError("Unknown potential candidate ID; call search first")
        return candidate
