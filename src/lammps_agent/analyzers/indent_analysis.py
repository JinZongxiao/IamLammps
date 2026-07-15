"""Deterministic nanoindentation analysis (Oliver-Pharr method).

Extracts hardness and reduced modulus from force-displacement data.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..thermo_parser import parse_thermo_table
from ._input import lammps_units
from .base import AnalyzerReport, BaseAnalyzer


class IndentAnalysisAnalyzer(BaseAnalyzer):
    name = "indent_analysis"

    def analyze(self, run_dir: Path) -> AnalyzerReport:
        stdout_path = run_dir / "stdout.log"
        if not stdout_path.exists():
            return AnalyzerReport(name=self.name, summary="No stdout.log", details={})

        thermo = parse_thermo_table(stdout_path)
        if lammps_units(stdout_path) != "metal":
            return AnalyzerReport(
                name=self.name,
                summary="Unsupported or missing LAMMPS units (indent_analysis requires metal)",
                details={"status": "unsupported_units"},
            )

        # Look for indenter force (f_indent_fix[3] = z-force) and position.
        # Column names depend on the fix name used in the script.
        # Try common patterns.
        force_col = None
        pos_col = None
        for col in thermo:
            col_lower = col.lower()
            if "indent" in col_lower and "[3]" in col:
                force_col = col
            if "indenter_z" in col_lower or "v_indenter" in col_lower:
                pos_col = col

        if force_col is None or len(thermo.get(force_col, [])) < 5:
            return AnalyzerReport(
                name=self.name,
                summary="Cannot find indenter force data in thermo output",
                details={"available_columns": list(thermo.keys())},
            )

        force = np.array(thermo[force_col], dtype=float)
        if not np.all(np.isfinite(force)):
            return AnalyzerReport(
                name=self.name,
                summary="Indenter force contains non-finite values",
                details={"status": "invalid_data"},
            )
        # Force should be in eV/Angstrom in metal units.
        # Convert to nN: 1 eV/Angstrom = 1.602e-9 N = 1.602 nN.
        force_nN = np.abs(force) * 1.602

        # If position data not available, use step as proxy for displacement.
        if pos_col and pos_col in thermo and len(thermo[pos_col]) == len(force):
            displacement = np.array(thermo[pos_col])
            displacement = displacement[0] - displacement  # indent depth
        else:
            return AnalyzerReport(
                name=self.name,
                summary="Cannot find explicit indenter displacement data",
                details={"status": "insufficient_data", "force_column": force_col},
            )
        if not np.all(np.isfinite(displacement)):
            return AnalyzerReport(
                name=self.name,
                summary="Indenter displacement contains non-finite values",
                details={"status": "invalid_data"},
            )

        # Oliver-Pharr analysis.
        P_max = float(np.max(force_nN))
        idx_max = int(np.argmax(force_nN))
        h_max = float(displacement[idx_max]) if idx_max < len(displacement) else 0.0

        # Simplified: estimate contact stiffness S from the slope near max load.
        S = None
        h_c = None
        hardness = None
        E_r = None

        if idx_max > 5:
            # Use last 20% of loading curve for stiffness estimate.
            start = max(0, idx_max - int(0.2 * idx_max))
            if displacement[idx_max] != displacement[start]:
                S = float(
                    (force_nN[idx_max] - force_nN[start])
                    / (displacement[idx_max] - displacement[start])
                )

        if S and S > 0 and P_max > 0:
            epsilon = 0.75  # geometric factor for Berkovich-like
            h_c = h_max - epsilon * P_max / S
            if h_c > 0:
                # Projected contact area (spherical approximation).
                # A = pi * (2*R*h_c - h_c^2) ≈ pi * 2*R*h_c for small h_c.
                # For simplicity, use A = 24.5 * h_c^2 (Berkovich equivalent).
                A = 24.5 * h_c ** 2
                if A > 0:
                    hardness = P_max / A  # nN / Angstrom^2 → GPa conversion
                    # 1 nN/A^2 = 1e-9 N / (1e-10 m)^2 = 10 GPa
                    hardness *= 10.0
                    E_r = math.sqrt(math.pi) * S / (2 * math.sqrt(A)) * 10.0

        summary_parts = [f"P_max={P_max:.2f} nN"]
        if hardness is not None:
            summary_parts.append(f"H={hardness:.2f} GPa")
        if E_r is not None:
            summary_parts.append(f"E_r={E_r:.2f} GPa")

        return AnalyzerReport(
            name=self.name,
            summary=", ".join(summary_parts),
            details={
                "P_max_nN": round(P_max, 4),
                "h_max_Angstrom": round(h_max, 4),
                "contact_stiffness_nN_per_A": round(S, 4) if S else None,
                "hardness_GPa": round(hardness, 4) if hardness else None,
                "reduced_modulus_GPa": round(E_r, 4) if E_r else None,
            },
        )
