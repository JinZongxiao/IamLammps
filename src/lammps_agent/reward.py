"""Reward scoring system for LAMMPS simulation runs.

Computes a multi-signal reward that quantifies how good a simulation run was.
Used by the memory system to prioritise high-quality experiences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .thermo_parser import parse_thermo_table


@dataclass
class RewardSignal:
    script_validity: float        # 0.0 or 1.0 — LAMMPS exited 0?
    convergence: float            # 0.0~1.0
    physical_plausibility: float  # 0.0~1.0
    efficiency: float             # 0.0~1.0 — penalty for retries
    total: float                  # weighted sum

    # Weights
    _W_VALID: float = 0.40
    _W_CONV: float = 0.25
    _W_PHYS: float = 0.25
    _W_EFF: float = 0.10


def compute_reward(
    return_code: int | None,
    run_dir: Path,
    retry_count: int = 0,
    task: str = "",
) -> RewardSignal:
    """Compute a composite reward signal from a completed run."""
    validity = 1.0 if return_code == 0 else 0.0
    efficiency = 1.0 / (1.0 + retry_count)

    # Try to parse thermo for convergence and physical plausibility.
    stdout_path = run_dir / "stdout.log"
    if stdout_path.exists() and validity > 0:
        thermo = parse_thermo_table(stdout_path)
        convergence = _score_convergence(thermo)
        plausibility = _score_physical_plausibility(thermo, task)
    else:
        convergence = 0.0
        plausibility = 0.0

    total = (
        RewardSignal._W_VALID * validity
        + RewardSignal._W_CONV * convergence
        + RewardSignal._W_PHYS * plausibility
        + RewardSignal._W_EFF * efficiency
    )
    return RewardSignal(
        script_validity=validity,
        convergence=convergence,
        physical_plausibility=plausibility,
        efficiency=efficiency,
        total=round(total, 4),
    )


# ---------------------------------------------------------------------------
# Internal scoring helpers
# ---------------------------------------------------------------------------

def _score_convergence(
    thermo: dict[str, list[float]], tail_fraction: float = 0.2
) -> float:
    """Score convergence from the TotEng (or PotEng) time series.

    Computes variance(last *tail_fraction*) / variance(full).
    Ratio < 0.1 → 1.0;  ratio > 1.0 → 0.0;  linear in between.
    """
    # Prefer TotEng, fall back to PotEng.
    series = thermo.get("TotEng") or thermo.get("PotEng") or []
    if len(series) < 5:
        return 0.5  # insufficient data — neutral score

    arr = np.array(series, dtype=np.float64)
    full_var = float(np.var(arr))
    if full_var == 0:
        return 1.0  # perfectly constant → converged

    tail_start = max(1, int(len(arr) * (1.0 - tail_fraction)))
    tail_var = float(np.var(arr[tail_start:]))
    ratio = tail_var / full_var

    if ratio <= 0.1:
        return 1.0
    if ratio >= 1.0:
        return 0.0
    return round(1.0 - (ratio - 0.1) / 0.9, 4)


def _score_physical_plausibility(
    thermo: dict[str, list[float]], task: str
) -> float:
    """Best-effort check that physical quantities are reasonable."""
    score = 1.0
    penalties = 0
    checks = 0

    # --- Temperature check ---
    temps = thermo.get("Temp", [])
    if temps:
        checks += 1
        final_temp = temps[-1]
        task_lower = task.lower()
        # Dynamics at finite T should have T > 0.
        is_dynamics = any(kw in task_lower for kw in ("nvt", "npt", "nve", "equilibrat", "动力学", "平衡"))
        is_minimize = any(kw in task_lower for kw in ("minimize", "最小化"))
        if is_dynamics and final_temp < 1e-10:
            penalties += 1  # T≈0 is suspicious for dynamics

    # --- Energy check: not NaN/Inf ---
    for col in ("TotEng", "PotEng"):
        series = thermo.get(col, [])
        if series:
            checks += 1
            if any(math.isnan(v) or math.isinf(v) for v in series):
                penalties += 1

    # --- For minimisation: energy should decrease ---
    if "minimize" in task.lower() or "最小化" in task.lower():
        pe = thermo.get("PotEng", [])
        if len(pe) >= 2:
            checks += 1
            if pe[-1] > pe[0]:
                penalties += 1  # energy increased

    if checks == 0:
        return 0.5  # no data to judge
    score = max(0.0, 1.0 - penalties / checks)
    return round(score, 4)
