"""Potential entry dataclass and registry of available interatomic potentials."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PotentialEntry:
    name: str                         # e.g. "W_Zhou2004_EAM"
    pair_style: str                   # e.g. "eam/alloy"
    elements: list[str]               # ["W"]
    file_path: Path                   # absolute path to potential file
    pair_coeff_template: str          # e.g. "pair_coeff * * {file_path} W"
    metadata: dict = field(default_factory=dict)
    lattice_type: str = "bcc"
    lattice_constant: float = 3.165
    atomic_masses: dict[str, float] = field(default_factory=dict)


class PotentialRegistry:
    """Registry of available interatomic potentials."""

    def __init__(self, potentials_dir: str | Path | None = None) -> None:
        self._entries: dict[str, PotentialEntry] = {}
        self._register_builtin(Path(potentials_dir) if potentials_dir else None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, name: str) -> PotentialEntry | None:
        return self._entries.get(name)

    def find_for_elements(self, elements: list[str]) -> list[PotentialEntry]:
        """Return all potentials that cover the given element set."""
        target = set(elements)
        return [e for e in self._entries.values() if target <= set(e.elements)]

    def register(self, entry: PotentialEntry) -> None:
        self._entries[entry.name] = entry

    def list_all(self) -> list[PotentialEntry]:
        return list(self._entries.values())

    # ------------------------------------------------------------------
    # Built-in potentials
    # ------------------------------------------------------------------

    def _register_builtin(self, configured_dir: Path | None = None) -> None:
        pot_dir = configured_dir or self._discover_lammps_potentials_dir()
        if pot_dir is None:
            return

        w_eam = pot_dir / "W_zhou.eam.alloy"
        if w_eam.exists():
            self.register(PotentialEntry(
                name="W_Zhou2004_EAM",
                pair_style="eam/alloy",
                elements=["W"],
                file_path=w_eam,
                pair_coeff_template="pair_coeff * * {file_path} W",
                metadata={
                    "source": "Zhou et al., Acta Mater 49 (2001) 4005",
                    "units": "metal",
                    "description": "EAM potential for BCC tungsten",
                },
                lattice_type="bcc",
                lattice_constant=3.165,
                atomic_masses={"W": 183.84},
            ))

    @staticmethod
    def _discover_lammps_potentials_dir() -> Path | None:
        """Try to find the LAMMPS bundled potentials directory.

        Strategy:
        1. ``LAMMPS_POTENTIALS_DIR`` environment variable.
        2. Relative to the ``lmp_mpi`` binary under ``../share/lammps/potentials``.
        3. Common Homebrew path (macOS).
        """
        import os

        env_dir = os.environ.get("LAMMPS_POTENTIALS_DIR")
        if env_dir:
            p = Path(env_dir)
            if p.is_dir():
                return p

        lmp = shutil.which("lmp_mpi") or shutil.which("lmp")
        if lmp:
            candidate = Path(lmp).resolve().parent.parent / "share" / "lammps" / "potentials"
            if candidate.is_dir():
                return candidate

        # Common Homebrew path on macOS.
        homebrew = Path("/opt/homebrew/share/lammps/potentials")
        if homebrew.is_dir():
            return homebrew

        return None
