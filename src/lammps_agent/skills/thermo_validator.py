"""Thermo validator skill.

Checks that physical quantities (temperature, energy, pressure) are
within reasonable ranges for the simulation.
"""

from __future__ import annotations

import math
from pathlib import Path

from ..thermo_parser import parse_thermo_table
from .base import BasePostprocessSkill, SkillReport


class ThermoValidatorSkill(BasePostprocessSkill):
    name = "thermo_validator"

    def analyze(self, run_dir: Path) -> SkillReport:
        stdout_path = run_dir / "stdout.log"
        if not stdout_path.exists():
            return SkillReport(
                name=self.name,
                summary="stdout.log missing",
                details={"status": "no_data"},
            )

        thermo = parse_thermo_table(stdout_path)
        warnings: list[str] = []

        # --- Temperature ---
        temps = thermo.get("Temp", [])
        if temps:
            final_temp = temps[-1]
            if final_temp < 0:
                warnings.append(f"Negative temperature detected: {final_temp}")
            if all(t < 1e-10 for t in temps) and len(temps) > 5:
                warnings.append(
                    "Temperature is essentially zero throughout the run — "
                    "likely missing 'velocity' command for dynamics"
                )

        # --- Energy: NaN / Inf ---
        for col in ("TotEng", "PotEng", "KinEng"):
            series = thermo.get(col, [])
            if series and any(math.isnan(v) or math.isinf(v) for v in series):
                warnings.append(f"{col} contains NaN or Inf values — simulation unstable")

        # --- Pressure: wildly divergent ---
        press = thermo.get("Press", [])
        if press:
            max_press = max(abs(p) for p in press)
            if max_press > 1e8:
                warnings.append(
                    f"Extreme pressure detected (max |P| = {max_press:.2e}) — "
                    "possible overlapping atoms or bad potential"
                )

        if not warnings:
            summary = "All physical quantities within reasonable ranges"
        else:
            summary = f"{len(warnings)} warning(s): " + "; ".join(warnings)

        return SkillReport(
            name=self.name,
            summary=summary,
            details={"warnings": warnings},
        )
