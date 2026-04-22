"""Shared base infrastructure for all workflow reporters.

Provides:
- _StabilityResult / _StageInfo dataclasses
- run_stability_checks()   — convergence + thermo_validator + T stability
- inventory_files()         — list key files in a run_dir
- write_report_files()      — write .md and .json, return paths
- warn / info helpers
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from ..skills.convergence import ConvergenceCheckSkill
from ..skills.thermo_validator import ThermoValidatorSkill
from ..thermo_parser import parse_thermo_table


# ---------------------------------------------------------------------------
# Shared dataclasses
# ---------------------------------------------------------------------------

@dataclass
class _StabilityResult:
    convergence_summary: str = "未提取"
    convergence_details: dict = field(default_factory=dict)
    thermo_validator_summary: str = "未提取"
    thermo_validator_warnings: list = field(default_factory=list)
    temperature_stable: Optional[bool] = None
    has_nan_inf: bool = False
    credibility: str = "未评估"


@dataclass
class _StageInfo:
    name: str
    run_dir: str
    return_code: Optional[int]
    retry_count: int
    success: bool


# ---------------------------------------------------------------------------
# Stability checks
# ---------------------------------------------------------------------------

def run_stability_checks(run_dir: Path, target_temperature: float = 0.0) -> _StabilityResult:
    """Run convergence + thermo_validator on *run_dir*."""
    s = _StabilityResult()

    try:
        conv = ConvergenceCheckSkill().analyze(run_dir)
        s.convergence_summary = conv.summary
        s.convergence_details = conv.details.get("convergence", {})
    except Exception as exc:
        s.convergence_summary = f"检查失败: {exc}"

    try:
        val = ThermoValidatorSkill().analyze(run_dir)
        s.thermo_validator_summary = val.summary
        s.thermo_validator_warnings = val.details.get("warnings", [])
        s.has_nan_inf = any(
            "NaN" in w or "Inf" in w for w in s.thermo_validator_warnings
        )
    except Exception as exc:
        s.thermo_validator_summary = f"检查失败: {exc}"

    # Temperature stability (if dynamics simulation)
    if target_temperature > 0:
        try:
            stdout_path = run_dir / "stdout.log"
            if stdout_path.exists():
                thermo = parse_thermo_table(stdout_path)
                temps = thermo.get("Temp", [])
                if temps and len(temps) >= 10:
                    tail = temps[int(len(temps) * 0.8):]
                    mean_t = float(np.mean(tail))
                    s.temperature_stable = abs(mean_t - target_temperature) / target_temperature < 0.20
        except Exception:
            pass

    # Credibility
    issues = []
    if s.has_nan_inf:
        issues.append("能量/压力含NaN/Inf")
    if s.thermo_validator_warnings:
        issues.append(f"{len(s.thermo_validator_warnings)}个物理量警告")
    if s.temperature_stable is False:
        issues.append("温度偏离目标值 >20%")

    s.credibility = (
        "结果可信（无物理异常，热力学量正常）"
        if not issues
        else f"结果可疑，存在以下问题: {'; '.join(issues)}"
    )
    return s


# ---------------------------------------------------------------------------
# File inventory
# ---------------------------------------------------------------------------

def inventory_files(run_dir: Path, extra_names: list[str] | None = None) -> list[str]:
    """Return list of key files present in *run_dir*."""
    candidates = [
        "in.lammps", "stdout.log", "stderr.log", "log.lammps", "result.json",
    ]
    # Auto-discover restart / dump files
    for p in sorted(run_dir.glob("restart.*")):
        candidates.append(p.name)
    for p in sorted(run_dir.glob("*.png")):
        candidates.append(p.name)
    for p in sorted(run_dir.glob("*.csv")):
        candidates.append(p.name)
    for p in sorted(run_dir.glob("*.json")):
        candidates.append(p.name)
    for p in sorted(run_dir.glob("in.lammps.retry*")):
        candidates.append(p.name)

    present = list(dict.fromkeys(f for f in candidates if (run_dir / f).exists()))

    # Dump files — just count them to keep list short
    single_dumps = sorted(run_dir.glob("dump.*.lammpstrj"))
    wildcard_dumps = sorted(run_dir.glob("dump.*.*.lammpstrj"))
    if single_dumps:
        for dp in single_dumps:
            if dp.name not in present:
                present.append(dp.name)
    if wildcard_dumps:
        present.append(f"dump.*.*.lammpstrj (共 {len(wildcard_dumps)} 帧)")

    if extra_names:
        for n in extra_names:
            if n not in present:
                present.append(n)

    return present


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def write_json(run_dir: Path, filename: str, payload: dict) -> Path:
    path = run_dir / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_md(run_dir: Path, filename: str, content: str) -> Path:
    path = run_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Stage info extraction
# ---------------------------------------------------------------------------

def extract_stage_infos(pipeline_result) -> list[_StageInfo]:
    from ..pipeline import PipelineResult  # avoid circular at module level
    infos = []
    for name, output in pipeline_result.stages:
        rc = output.lammps_result.return_code if output.lammps_result else None
        infos.append(_StageInfo(
            name=name,
            run_dir=str(output.run_dir),
            return_code=rc,
            retry_count=output.retry_count,
            success=(rc == 0),
        ))
    return infos


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def report_info(msg: str) -> None:
    print(f"[REPORT] {msg}", file=sys.stderr)


def report_warn(msg: str) -> None:
    warnings.warn(f"[REPORT] {msg}", stacklevel=3)
    print(f"[REPORT WARNING] {msg}", file=sys.stderr)
