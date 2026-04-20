"""Convergence checking skill.

Checks whether energy and temperature have converged by comparing
tail-region variance to full-series variance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..thermo_parser import parse_thermo_table
from .base import BasePostprocessSkill, SkillReport


class ConvergenceCheckSkill(BasePostprocessSkill):
    name = "convergence"

    def analyze(self, run_dir: Path) -> SkillReport:
        stdout_path = run_dir / "stdout.log"
        if not stdout_path.exists():
            return SkillReport(
                name=self.name,
                summary="stdout.log missing",
                details={"status": "no_data"},
            )

        thermo = parse_thermo_table(stdout_path)
        results: dict[str, str] = {}

        for col in ("TotEng", "Temp"):
            series = thermo.get(col, [])
            if len(series) < 10:
                results[col] = "insufficient_data"
                continue

            arr = np.array(series, dtype=np.float64)
            full_var = float(np.var(arr))
            if full_var == 0:
                results[col] = "converged"
                continue

            tail_start = max(1, int(len(arr) * 0.8))
            tail_var = float(np.var(arr[tail_start:]))
            ratio = tail_var / full_var

            if ratio <= 0.15:
                results[col] = "converged"
            elif ratio <= 0.5:
                results[col] = "partially_converged"
            else:
                results[col] = "not_converged"

        statuses = list(results.values())
        if all(s == "converged" for s in statuses):
            summary = "All monitored quantities converged"
        elif any(s == "not_converged" for s in statuses):
            bad = [k for k, v in results.items() if v == "not_converged"]
            summary = f"Not converged: {', '.join(bad)}"
        else:
            summary = "Partially converged"

        return SkillReport(
            name=self.name,
            summary=summary,
            details={"convergence": results},
        )
