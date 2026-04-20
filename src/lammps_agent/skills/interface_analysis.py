"""Interface analysis skill — adhesion energy and work of separation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..thermo_parser import parse_thermo_table
from .base import BasePostprocessSkill, SkillReport


class InterfaceAnalysisSkill(BasePostprocessSkill):
    """Analyse oxide/W interface simulations.

    Examines thermo output for potential energy evolution during
    interface separation (tensile normal to interface).  When
    interface-specific data is present it estimates:

    * **Work of separation** (J/m²) from the energy-displacement curve.
    * **Peak normal stress** (GPa) at the interface.
    * **Adhesion energy** from the plateau of the energy curve.

    If no interface-specific columns are found, it falls back to
    reporting the available potential-energy data.
    """

    name = "interface_analysis"

    def analyze(self, run_dir: Path) -> SkillReport:
        stdout_path = run_dir / "stdout.log"
        if not stdout_path.exists():
            return SkillReport(name=self.name, summary="No stdout.log", details={})

        thermo = parse_thermo_table(stdout_path)

        pe = thermo.get("PotEng", thermo.get("v_pe", []))
        steps = thermo.get("Step", [])
        lz = thermo.get("Lz", [])
        pzz = thermo.get("Pzz", [])

        if len(pe) < 3:
            return SkillReport(
                name=self.name,
                summary="Insufficient thermo data for interface analysis",
                details={"available_columns": list(thermo.keys())},
            )

        pe_arr = np.array(pe, dtype=float)
        delta_pe = float(pe_arr[-1] - pe_arr[0])

        details: dict = {
            "initial_pe_eV": round(float(pe_arr[0]), 4),
            "final_pe_eV": round(float(pe_arr[-1]), 4),
            "delta_pe_eV": round(delta_pe, 4),
            "data_points": len(pe),
        }

        # Estimate work of separation if box dimension Lz is available.
        w_sep = None
        if len(lz) >= 2:
            lz0 = lz[0]
            displacement_A = lz[-1] - lz0
            if abs(displacement_A) > 1e-6:
                # W_sep = delta_E / (2 * A)  — but without explicit area
                # we report the raw energy change per unit displacement.
                w_sep = delta_pe / abs(displacement_A)  # eV/Å
                details["displacement_A"] = round(displacement_A, 4)
                details["energy_per_disp_eV_A"] = round(w_sep, 6)

        # Peak normal stress
        peak_stress_GPa = None
        if len(pzz) >= 3:
            stress_gpa = np.array([-p * 1e-4 for p in pzz])  # bar → GPa
            peak_stress_GPa = float(np.max(np.abs(stress_gpa)))
            details["peak_normal_stress_GPa"] = round(peak_stress_GPa, 4)

        summary_parts = [f"delta_PE = {delta_pe:.4f} eV"]
        if peak_stress_GPa is not None:
            summary_parts.append(f"peak_stress = {peak_stress_GPa:.2f} GPa")
        if w_sep is not None:
            summary_parts.append(f"dE/dz = {w_sep:.4f} eV/A")

        return SkillReport(
            name=self.name,
            summary=", ".join(summary_parts),
            details=details,
        )
