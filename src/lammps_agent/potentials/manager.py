"""Potential file management — copy/symlink, command generation."""

from __future__ import annotations

import shutil
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

    def prepare_for_run(self, potential: PotentialEntry, run_dir: Path) -> Path:
        """Copy the potential file into *run_dir* and return the local path.

        Copying (rather than symlinking) ensures portability to compute nodes
        where the original path may not be accessible.
        """
        dest = run_dir / potential.file_path.name
        if not dest.exists():
            shutil.copy2(potential.file_path, dest)
        return dest

    def generate_pair_commands(
        self, potential: PotentialEntry, local_path: Path | None = None
    ) -> str:
        """Return ``pair_style`` + ``pair_coeff`` LAMMPS commands."""
        file_ref = str(local_path.name) if local_path else str(potential.file_path)
        pair_coeff = potential.pair_coeff_template.replace("{file_path}", file_ref)
        return f"pair_style {potential.pair_style}\n{pair_coeff}"

    def generate_full_setup(self, potential: PotentialEntry, local_path: Path | None = None) -> str:
        """Return a full setup block: units, atom_style, lattice, mass, pair commands."""
        file_ref = str(local_path.name) if local_path else str(potential.file_path)
        pair_coeff = potential.pair_coeff_template.replace("{file_path}", file_ref)

        lines = [
            "units metal",
            "atom_style atomic",
            "boundary p p p",
            "",
            f"lattice {potential.lattice_type} {potential.lattice_constant}",
        ]

        # Mass commands for all elements.
        for i, elem in enumerate(potential.elements, start=1):
            mass = potential.atomic_masses.get(elem, 1.0)
            lines.append(f"# mass {i} {mass}  # {elem}")

        lines.append("")
        lines.append(f"pair_style {potential.pair_style}")
        lines.append(pair_coeff)

        return "\n".join(lines)
