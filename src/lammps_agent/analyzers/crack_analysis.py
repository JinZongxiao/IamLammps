"""Deterministic crack tip and fracture analysis."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..thermo_parser import parse_thermo_table
from ._input import lammps_units
from .base import AnalyzerReport, BaseAnalyzer


class CrackAnalysisAnalyzer(BaseAnalyzer):
    name = "crack_analysis"

    def analyze(self, run_dir: Path) -> AnalyzerReport:
        stdout_path = run_dir / "stdout.log"
        if not stdout_path.exists():
            return AnalyzerReport(name=self.name, summary="No stdout.log", details={})

        thermo = parse_thermo_table(stdout_path)
        if lammps_units(stdout_path) != "metal":
            return AnalyzerReport(
                name=self.name,
                summary="Unsupported or missing LAMMPS units (crack_analysis requires metal)",
                details={"status": "unsupported_units"},
            )

        # Extract stress (Pyy for Mode I in y-direction) and box dimension Ly.
        pyy = thermo.get("Pyy", [])
        ly = thermo.get("Ly", [])
        steps = thermo.get("Step", [])

        if len(pyy) < 5:
            return AnalyzerReport(
                name=self.name,
                summary="Insufficient data for crack analysis (need Pyy column)",
                details={"available_columns": list(thermo.keys())},
            )
        if not all(math.isfinite(value) for value in pyy):
            return AnalyzerReport(
                name=self.name,
                summary="Crack stress data contains non-finite values",
                details={"status": "invalid_data"},
            )
        if steps and len(steps) != len(pyy):
            return AnalyzerReport(
                name=self.name,
                summary="Crack stress and step columns are misaligned",
                details={"status": "invalid_data"},
            )

        # Convert stress to GPa (LAMMPS metal: bars → GPa).
        stress_gpa = np.array([-p * 1e-4 for p in pyy])

        max_stress = float(np.max(stress_gpa))
        max_stress_step = int(steps[np.argmax(stress_gpa)]) if steps else 0

        # Strain if Ly available.
        fracture_strain = None
        if len(ly) == len(pyy) and len(ly) >= 2 and all(math.isfinite(value) for value in ly):
            ly0 = ly[0]
            if ly0 != 0:
                strain = [(length - ly0) / ly0 for length in ly]
                fracture_strain = round(float(max(strain)), 4)

        # Simple K_IC estimation: K = sigma * sqrt(pi * a)
        # Requires knowledge of initial crack length (not directly in thermo).
        # Report max stress for now; actual K_IC needs dump file post-processing.

        summary = f"Max stress (Mode I) = {max_stress:.2f} GPa at step {max_stress_step}"
        if fracture_strain:
            summary += f", max strain = {fracture_strain:.4f}"

        return AnalyzerReport(
            name=self.name,
            summary=summary,
            details={
                "max_stress_GPa": round(max_stress, 4),
                "max_stress_step": max_stress_step,
                "fracture_strain": fracture_strain,
                "data_points": len(pyy),
            },
        )
