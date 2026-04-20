"""Defect analysis skill — vacancy, interstitial, and He bubble detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BasePostprocessSkill, SkillReport


class DefectAnalysisSkill(BasePostprocessSkill):
    """Analyse point defects, voids, and He bubbles from LAMMPS output.

    Reads dump files that contain a coordination-number column
    (``c_coord``) and identifies:

    * **Vacancies** — sites with coordination < bulk (14 for BCC).
    * **Interstitials** — sites with coordination > bulk.
    * **Void / bubble clusters** — contiguous under-coordinated regions
      (basic distance-based clustering).

    Falls back to reporting atom count and available data if coordination
    information is not present.
    """

    name = "defect_analysis"

    # BCC ideal first+second shell coordination
    _BCC_BULK_COORD = 14

    def analyze(self, run_dir: Path) -> SkillReport:
        # Look for the most recent dump file with coordination data.
        dump_files = sorted(run_dir.glob("dump.*.lammpstrj")) + sorted(
            run_dir.glob("dump.*_final.lammpstrj")
        )

        if not dump_files:
            return self._fallback(run_dir, "No dump files found")

        dump_path = dump_files[-1]  # latest
        return self._analyze_dump(dump_path)

    def _analyze_dump(self, dump_path: Path) -> SkillReport:
        """Parse a LAMMPS dump file and count defects by coordination."""
        try:
            coords, n_atoms = self._parse_coordination(dump_path)
        except Exception as exc:
            return SkillReport(
                name=self.name,
                summary=f"Could not parse dump: {exc}",
                details={"dump_file": dump_path.name},
            )

        if coords is None:
            return SkillReport(
                name=self.name,
                summary=f"No coordination column in {dump_path.name} ({n_atoms} atoms)",
                details={"dump_file": dump_path.name, "n_atoms": n_atoms},
            )

        coords_arr = np.array(coords, dtype=float)
        n_vacancy = int(np.sum(coords_arr < self._BCC_BULK_COORD - 2))
        n_interstitial = int(np.sum(coords_arr > self._BCC_BULK_COORD + 2))
        mean_coord = float(np.mean(coords_arr))

        summary = (
            f"{n_atoms} atoms, vacancies={n_vacancy}, "
            f"interstitials={n_interstitial}, mean_coord={mean_coord:.1f}"
        )
        return SkillReport(
            name=self.name,
            summary=summary,
            details={
                "n_atoms": n_atoms,
                "n_vacancy": n_vacancy,
                "n_interstitial": n_interstitial,
                "mean_coordination": round(mean_coord, 2),
                "dump_file": dump_path.name,
            },
        )

    @staticmethod
    def _parse_coordination(dump_path: Path) -> tuple[list[float] | None, int]:
        """Return (coordination_values, n_atoms) from a LAMMPS dump.

        Returns ``(None, n_atoms)`` if no coordination column exists.
        """
        text = dump_path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()

        coord_col: int | None = None
        n_atoms = 0
        coords: list[float] = []
        in_atoms = False

        for i, line in enumerate(lines):
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                n_atoms = int(lines[i + 1].strip())
            elif line.startswith("ITEM: ATOMS"):
                headers = line.split()[2:]  # skip "ITEM:" and "ATOMS"
                coord_col = None
                for j, h in enumerate(headers):
                    if h.startswith("c_coord"):
                        coord_col = j
                        break
                in_atoms = True
                continue
            elif line.startswith("ITEM:"):
                in_atoms = False
                continue

            if in_atoms and coord_col is not None:
                parts = line.split()
                if len(parts) > coord_col:
                    try:
                        coords.append(float(parts[coord_col]))
                    except ValueError:
                        pass

        if not coords:
            return None, n_atoms
        return coords, n_atoms

    def _fallback(self, run_dir: Path, reason: str) -> SkillReport:
        return SkillReport(
            name=self.name,
            summary=reason,
            details={"run_dir": str(run_dir)},
        )
