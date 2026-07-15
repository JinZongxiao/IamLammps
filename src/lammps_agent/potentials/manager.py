"""Potential file management — copy/symlink, command generation."""

from __future__ import annotations

from pathlib import Path

from .registry import PotentialEntry, PotentialRegistry


class PotentialManager:
    """Manage potential files for simulation runs."""

    def __init__(self, registry: PotentialRegistry | None = None) -> None:
        self.registry = registry or PotentialRegistry()

    def resolve_potential(self, name: str) -> PotentialEntry:
        """Get a potential entry by name, raising if not found."""
        entry = self.registry.get(name)
        if entry is None:
            available = [e.name for e in self.registry.list_all()]
            raise ValueError(
                f"Potential '{name}' not found. Available: {available}"
            )
        return entry

    def generate_pair_commands(
        self, potential: PotentialEntry, local_path: Path | None = None
    ) -> str:
        """Return ``pair_style`` + ``pair_coeff`` LAMMPS commands."""
        file_ref = str(local_path.name) if local_path else str(potential.file_path)
        pair_coeff = potential.pair_coeff_template.replace("{file_path}", file_ref)
        return f"pair_style {potential.pair_style}\n{pair_coeff}"
