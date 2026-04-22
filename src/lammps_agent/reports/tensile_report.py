"""Stress-strain curve analysis skill.

Extracts mechanical properties from LAMMPS tensile simulation output:
- Young's modulus (linear fit at small strain)
- Yield strength (0.2% offset method)
- Ultimate tensile strength (max stress)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..thermo_parser import parse_thermo_table
from .base import BasePostprocessSkill, SkillReport


class StressStrainSkill(BasePostprocessSkill):
    name = "stress_strain"

    def analyze(self, run_dir: Path) -> SkillReport:
        stdout_path = run_dir / "stdout.log"
        if not stdout_path.exists():
            return SkillReport(name=self.name, summary="No stdout.log", details={})

        thermo = parse_thermo_table(stdout_path)

        # Need box dimensions (lz) and stress (pzz) for z-axis tensile.
        lz = thermo.get("Lz", [])
        pzz = thermo.get("Pzz", [])

        if len(lz) < 5 or len(pzz) < 5:
            return SkillReport(
                name=self.name,
                summary="Insufficient stress/strain data (need Lz and Pzz columns)",
                details={"available_columns": list(thermo.keys())},
            )

        # Compute engineering strain and stress.
        lz0 = lz[0]
        strain = [(l - lz0) / lz0 for l in lz]
        # LAMMPS pressure is in bars (metal units). Convert to GPa: 1 bar = 1e-4 GPa.
        # Stress = -Pzz (compressive convention in LAMMPS).
        stress_gpa = [-p * 1e-4 for p in pzz]

        strain_arr = np.array(strain)
        stress_arr = np.array(stress_gpa)

        # Young's modulus: linear fit for strain < 0.02.
        elastic_mask = (strain_arr > 0) & (strain_arr < 0.02)
        youngs_modulus = None
        if np.sum(elastic_mask) >= 3:
            coeffs = np.polyfit(strain_arr[elastic_mask], stress_arr[elastic_mask], 1)
            youngs_modulus = round(float(coeffs[0]), 2)  # GPa

        # UTS: max stress.
        uts = round(float(np.max(stress_arr)), 4) if len(stress_arr) > 0 else None

        # Yield strength: 0.2% offset method.
        yield_strength = None
        if youngs_modulus is not None and youngs_modulus > 0:
            offset_line = youngs_modulus * (strain_arr - 0.002)
            diff = stress_arr - offset_line
            # Find where stress drops below the offset line (sign change).
            sign_changes = np.where(np.diff(np.sign(diff)))[0]
            if len(sign_changes) > 0:
                idx = sign_changes[0]
                yield_strength = round(float(stress_arr[idx]), 4)

        summary_parts = []
        if youngs_modulus is not None:
            summary_parts.append(f"E={youngs_modulus:.1f} GPa")
        if yield_strength is not None:
            summary_parts.append(f"σ_y={yield_strength:.2f} GPa")
        if uts is not None:
            summary_parts.append(f"UTS={uts:.2f} GPa")

        return SkillReport(
            name=self.name,
            summary=", ".join(summary_parts) if summary_parts else "Could not extract properties",
            details={
                "youngs_modulus_GPa": youngs_modulus,
                "yield_strength_GPa": yield_strength,
                "ultimate_tensile_strength_GPa": uts,
                "data_points": len(strain),
                "max_strain": round(float(max(strain)), 4) if strain else 0,
            },
        )
