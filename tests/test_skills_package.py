"""Tests for deterministic analyzers."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_analyzers_import():
    from lammps_agent.analyzers import (
        _ANALYZER_REGISTRY,
    )
    assert "thermo_tail" in _ANALYZER_REGISTRY
    assert "convergence" in _ANALYZER_REGISTRY
    assert "thermo_validator" in _ANALYZER_REGISTRY


def test_thermo_tail_skill(successful_run_dir: Path):
    from lammps_agent.analyzers import run_analyzers

    reports = run_analyzers(successful_run_dir, ["thermo_tail"])
    assert len(reports) == 1
    assert reports[0].name == "thermo_tail"
    assert reports[0].details["thermo_rows"] > 0


def test_convergence_skill(successful_run_dir: Path):
    from lammps_agent.analyzers import run_analyzers

    reports = run_analyzers(successful_run_dir, ["convergence"])
    assert len(reports) == 1
    assert reports[0].name == "convergence"
    assert "converge" in reports[0].summary.lower() or "data" in reports[0].summary.lower()


def test_convergence_skill_detects_sustained_drift(tmp_path: Path):
    from lammps_agent.analyzers import run_analyzers

    run_dir = tmp_path / "drifting"
    run_dir.mkdir()
    rows = "\n".join(
        f"{index} {100.0 - index * 0.2} {300.0 + index * 0.1}"
        for index in range(20)
    )
    (run_dir / "stdout.log").write_text(
        f"Step TotEng Temp\n{rows}\nLoop time of 1.0 on 1 procs\n",
        encoding="utf-8",
    )

    report = run_analyzers(run_dir, ["convergence"])[0]

    assert report.details["convergence"] == {
        "TotEng": "not_converged",
        "Temp": "not_converged",
    }
    assert report.details["sustained_drift"] == {"TotEng": True, "Temp": True}
    assert "sustained drift" in report.summary


def test_thermo_validator_normal(successful_run_dir: Path):
    from lammps_agent.analyzers import run_analyzers

    reports = run_analyzers(successful_run_dir, ["thermo_validator"])
    assert len(reports) == 1
    assert reports[0].name == "thermo_validator"


def test_thermo_validator_zero_temp(zero_temp_run_dir: Path):
    from lammps_agent.analyzers import run_analyzers

    reports = run_analyzers(zero_temp_run_dir, ["thermo_validator"])
    assert len(reports) == 1
    # Should flag zero temperature.
    assert "zero" in reports[0].summary.lower() or "warning" in reports[0].summary.lower()


def test_all_skills_together(successful_run_dir: Path):
    from lammps_agent.analyzers import run_analyzers

    reports = run_analyzers(
        successful_run_dir, ["thermo_tail", "convergence", "thermo_validator"]
    )
    assert len(reports) == 3
    names = {r.name for r in reports}
    assert names == {"thermo_tail", "convergence", "thermo_validator"}


def test_unknown_analyzer_is_not_silently_ignored(successful_run_dir: Path):
    from lammps_agent.analyzers import UnknownAnalyzerError, run_analyzers

    with pytest.raises(UnknownAnalyzerError) as raised:
        run_analyzers(successful_run_dir, ["does-not-exist"])

    assert raised.value.as_dict()["code"] == "UNKNOWN_ANALYZER"
