"""Shared validation helpers for deterministic analyzers."""

from __future__ import annotations

from pathlib import Path


def lammps_units(path: Path) -> str | None:
    """Return an explicit ``units`` declaration captured with analyzer input."""

    declared = _declared_units(path)
    workspace = path.parent / "workspace"
    if workspace.is_dir() and not workspace.is_symlink():
        candidates = sorted(
            candidate
            for candidate in workspace.rglob("*")
            if candidate.is_file()
            and not candidate.is_symlink()
            and (
                candidate.name.casefold().startswith("in.")
                or candidate.suffix.casefold() in {".in", ".lmp", ".lammps"}
            )
        )
        for candidate in candidates[:100]:
            declared.update(_declared_units(candidate))
    return next(iter(declared)) if len(declared) == 1 else None


def _declared_units(path: Path) -> set[str]:
    values: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        tokens = line.strip().split()
        if len(tokens) == 2 and tokens[0].casefold() == "units":
            values.add(tokens[1].casefold())
    return values
