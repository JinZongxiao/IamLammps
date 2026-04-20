"""Post-processing skills package.

Re-exports the public API so that ``from .skills import SkillReport, run_postprocess_skills``
continues to work unchanged in agent.py after the refactoring from a single file to a package.
"""

from __future__ import annotations

from pathlib import Path

from .base import BasePostprocessSkill, SkillReport
from .thermo_tail import ThermoTailSkill
from .convergence import ConvergenceCheckSkill
from .thermo_validator import ThermoValidatorSkill
from .stress_strain import StressStrainSkill
from .indent_analysis import IndentAnalysisSkill
from .crack_analysis import CrackAnalysisSkill
from .dbtt_fit import DBTTFitSkill
from .interface_analysis import InterfaceAnalysisSkill
from .defect_analysis import DefectAnalysisSkill

__all__ = [
    "BasePostprocessSkill",
    "SkillReport",
    "ThermoTailSkill",
    "ConvergenceCheckSkill",
    "ThermoValidatorSkill",
    "StressStrainSkill",
    "IndentAnalysisSkill",
    "CrackAnalysisSkill",
    "DBTTFitSkill",
    "InterfaceAnalysisSkill",
    "DefectAnalysisSkill",
    "build_skills",
    "run_postprocess_skills",
]

_SKILL_REGISTRY: dict[str, type[BasePostprocessSkill]] = {
    ThermoTailSkill.name: ThermoTailSkill,
    ConvergenceCheckSkill.name: ConvergenceCheckSkill,
    ThermoValidatorSkill.name: ThermoValidatorSkill,
    StressStrainSkill.name: StressStrainSkill,
    IndentAnalysisSkill.name: IndentAnalysisSkill,
    CrackAnalysisSkill.name: CrackAnalysisSkill,
    DBTTFitSkill.name: DBTTFitSkill,
    InterfaceAnalysisSkill.name: InterfaceAnalysisSkill,
    DefectAnalysisSkill.name: DefectAnalysisSkill,
}


def register_skill(cls: type[BasePostprocessSkill]) -> None:
    """Register an additional skill at runtime."""
    _SKILL_REGISTRY[cls.name] = cls


def build_skills(enabled: list[str]) -> list[BasePostprocessSkill]:
    skills: list[BasePostprocessSkill] = []
    for name in enabled:
        cls = _SKILL_REGISTRY.get(name)
        if cls is not None:
            skills.append(cls())
    return skills


def run_postprocess_skills(run_dir: Path, enabled: list[str]) -> list[SkillReport]:
    reports: list[SkillReport] = []
    for skill in build_skills(enabled):
        try:
            reports.append(skill.analyze(run_dir))
        except Exception as exc:
            reports.append(
                SkillReport(
                    name=skill.name,
                    summary=f"Skill failed: {exc}",
                    details={"error": str(exc)},
                )
            )
    return reports
