"""Tests for the skills package (import + individual skills)."""

from __future__ import annotations

from pathlib import Path


def test_skills_import():
    """The skills package should export the same API as the old skills.py."""
    from src.lammps_agent.skills import (
        SkillReport,
        BasePostprocessSkill,
        run_postprocess_skills,
        build_skills,
        _SKILL_REGISTRY,
    )
    assert "thermo_tail" in _SKILL_REGISTRY
    assert "convergence" in _SKILL_REGISTRY
    assert "thermo_validator" in _SKILL_REGISTRY


def test_thermo_tail_skill(successful_run_dir: Path):
    from src.lammps_agent.skills import run_postprocess_skills

    reports = run_postprocess_skills(successful_run_dir, ["thermo_tail"])
    assert len(reports) == 1
    assert reports[0].name == "thermo_tail"
    assert reports[0].details["thermo_rows"] > 0


def test_convergence_skill(successful_run_dir: Path):
    from src.lammps_agent.skills import run_postprocess_skills

    reports = run_postprocess_skills(successful_run_dir, ["convergence"])
    assert len(reports) == 1
    assert reports[0].name == "convergence"
    assert "converge" in reports[0].summary.lower() or "data" in reports[0].summary.lower()


def test_thermo_validator_normal(successful_run_dir: Path):
    from src.lammps_agent.skills import run_postprocess_skills

    reports = run_postprocess_skills(successful_run_dir, ["thermo_validator"])
    assert len(reports) == 1
    assert reports[0].name == "thermo_validator"


def test_thermo_validator_zero_temp(zero_temp_run_dir: Path):
    from src.lammps_agent.skills import run_postprocess_skills

    reports = run_postprocess_skills(zero_temp_run_dir, ["thermo_validator"])
    assert len(reports) == 1
    # Should flag zero temperature.
    assert "zero" in reports[0].summary.lower() or "warning" in reports[0].summary.lower()


def test_all_skills_together(successful_run_dir: Path):
    from src.lammps_agent.skills import run_postprocess_skills

    reports = run_postprocess_skills(
        successful_run_dir, ["thermo_tail", "convergence", "thermo_validator"]
    )
    assert len(reports) == 3
    names = {r.name for r in reports}
    assert names == {"thermo_tail", "convergence", "thermo_validator"}
