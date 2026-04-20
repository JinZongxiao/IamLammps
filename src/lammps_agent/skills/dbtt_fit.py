"""DBTT fitting skill — Boltzmann sigmoid fit to temperature vs ductility."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BasePostprocessSkill, SkillReport


class DBTTFitSkill(BasePostprocessSkill):
    name = "dbtt_fit"

    def analyze(self, run_dir: Path) -> SkillReport:
        """Basic placeholder — actual DBTT fitting requires aggregated data
        from multiple temperature runs, which is handled by analyze_sweep().
        """
        return SkillReport(
            name=self.name,
            summary="Use analyze_sweep() with temperature-ductility data",
            details={},
        )

    def analyze_sweep(
        self,
        temperatures: list[float],
        ductilities: list[float],
    ) -> SkillReport:
        """Fit a Boltzmann sigmoid to temperature-ductility data.

        Parameters
        ----------
        temperatures : list of T values (K).
        ductilities : list of fracture strain (or elongation) at each T.
        """
        if len(temperatures) < 3 or len(ductilities) < 3:
            return SkillReport(
                name=self.name,
                summary="Insufficient data for DBTT fit (need >= 3 temperature points)",
                details={"n_points": len(temperatures)},
            )

        T_arr = np.array(temperatures, dtype=np.float64)
        d_arr = np.array(ductilities, dtype=np.float64)

        # Try to fit Boltzmann sigmoid: f(T) = d_min + (d_max - d_min) / (1 + exp(-(T - T_dbtt) / w))
        T_dbtt = None
        width = None
        try:
            from scipy.optimize import curve_fit

            def sigmoid(T: np.ndarray, T0: float, w: float, d_min: float, d_max: float) -> np.ndarray:
                return d_min + (d_max - d_min) / (1.0 + np.exp(-(T - T0) / w))

            # Initial guesses.
            p0 = [
                float(np.median(T_arr)),  # T_dbtt
                100.0,                    # width
                float(np.min(d_arr)),     # d_min
                float(np.max(d_arr)),     # d_max
            ]
            popt, _ = curve_fit(sigmoid, T_arr, d_arr, p0=p0, maxfev=5000)
            T_dbtt = round(float(popt[0]), 1)
            width = round(float(abs(popt[1])), 1)
        except Exception:
            # Fall back to simple interpolation if scipy fit fails.
            mid_ductility = (float(np.min(d_arr)) + float(np.max(d_arr))) / 2.0
            # Find temperature closest to mid-ductility.
            idx = int(np.argmin(np.abs(d_arr - mid_ductility)))
            T_dbtt = round(float(T_arr[idx]), 1)

        summary = f"DBTT ≈ {T_dbtt} K" if T_dbtt else "Could not determine DBTT"
        if width:
            summary += f" (transition width ≈ {width} K)"

        return SkillReport(
            name=self.name,
            summary=summary,
            details={
                "T_dbtt_K": T_dbtt,
                "transition_width_K": width,
                "temperatures": [round(t, 1) for t in temperatures],
                "ductilities": [round(d, 4) for d in ductilities],
            },
        )
